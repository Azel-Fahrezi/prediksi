from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import uuid
import pickle

app = Flask(__name__)
app.secret_key = 'secretkey123'
UPLOAD_FOLDER = 'uploads'
SAVE_FOLDER = 'saved_models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred) * 100
    return {"RMSE": rmse, "MSE": mse, "MAPE": mape, "R2": r2}

def train_model(dataframe, model_type, max_iter, target_accuracy):
    encoder = LabelEncoder()
    dataframe['Musim_Periodik'] = encoder.fit_transform(dataframe['Musim_Periodik'])
    label_mapping = {str(k): int(v) for k, v in zip(encoder.classes_, range(len(encoder.classes_)))}
    original_labels = list(encoder.classes_)

    X = dataframe[['Jumlah_Stok', 'Musim_Periodik', 'Penjualan_Sebelumnya']].values
    y = dataframe['Penjualan_Prediksi'].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3333, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "SVM":
        model = SVR(kernel='rbf', max_iter=max_iter)
    elif model_type == "ANN":
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, learning_rate_init=0.001)

    model.fit(X_train_scaled, y_train)

    train_metrics = calculate_metrics(y_train, model.predict(X_train_scaled))
    val_metrics = calculate_metrics(y_val, model.predict(X_val_scaled))

    accuracy = val_metrics["R2"]
    if accuracy >= target_accuracy:
        result = "Model telah mencapai target akurasi."
    else:
        result = "Model belum mencapai target akurasi."

    metrics = {
        model_type: {
            "Training Metrics": train_metrics,
            "Validation Metrics": val_metrics,
            "Result": result
        }
    }

    return model, scaler, label_mapping, metrics, original_labels

def predict_model(model, scaler, label_mapping, features):
    encoder = LabelEncoder()
    encoder.classes_ = np.array(session.get('original_labels', []))

    # Validasi input
    if features['Musim_Periodik'] not in encoder.classes_:
        raise ValueError(f"Nilai 'Musim_Periodik' tidak valid. Diperbolehkan: {list(encoder.classes_)}")

    # Transformasi label
    Musim_Periodik_encoded = encoder.transform([features['Musim_Periodik']])[0]
    input_features = np.array([[features['Jumlah_Stok'], Musim_Periodik_encoded, features['Penjualan_Sebelumnya']]])
    input_scaled = scaler.transform(input_features)

    return model.predict(input_scaled)[0]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            required_columns = ['Jumlah_Stok', 'Musim_Periodik', 'Penjualan_Sebelumnya', 'Penjualan_Prediksi']
            if not all(col in df.columns for col in required_columns):
                os.remove(filepath)
                return f"Kolom yang diperlukan tidak ditemukan. Pastikan file memiliki kolom: {', '.join(required_columns)}."

            session['uploaded_file'] = filepath
            session['data_shape'] = df.shape
            session['current_page'] = 1

            # Latih model dan simpan metrik
            svm_model, svm_scaler, svm_label_mapping, metrics_svm, svm_labels = train_model(df, "SVM", max_iter=500, target_accuracy=95)
            ann_model, ann_scaler, ann_label_mapping, metrics_ann, ann_labels = train_model(df, "ANN", max_iter=500, target_accuracy=95)

            session['metrics_svm'] = metrics_svm
            session['metrics_ann'] = metrics_ann

            return redirect(url_for('index'))
        except Exception as e:
            os.remove(filepath)
            return f"Error membaca file: {str(e)}"

    filepath = session.get('uploaded_file')
    if filepath and not os.path.exists(filepath):
        # File tidak ditemukan, reset sesi
        session.pop('uploaded_file', None)
        session.pop('data_shape', None)
        session.pop('current_page', None)
        session.pop('metrics_svm', None)
        session.pop('metrics_ann', None)
        return redirect(url_for('index'))

    if filepath:
        df = pd.read_csv(filepath)
        rows_per_page = 10
        page = session.get('current_page', 1)
        start_row = (page - 1) * rows_per_page
        end_row = start_row + rows_per_page
        page_data = df.iloc[start_row:end_row]
        table_html = page_data.to_html(classes="table table-bordered table-striped", index=False)

        metrics_svm = session.get('metrics_svm')
        metrics_ann = session.get('metrics_ann')

        return render_template(
            "combined.html",
            file_uploaded=True,
            table_html=table_html,
            filename=os.path.basename(filepath),
            page=page,
            has_prev=page > 1,
            has_next=end_row < len(df),
            metrics_svm=metrics_svm,
            metrics_ann=metrics_ann
        )
    return render_template("combined.html", file_uploaded=False)

@app.route("/testing", methods=["GET", "POST"])
def testing():
    return render_template("testing.html", message="Halaman Testing")

@app.route("/paginate/<int:page>", methods=["GET"])
def paginate(page):
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    df = pd.read_csv(filepath)
    rows_per_page = 10
    total_pages = (len(df) + rows_per_page - 1) // rows_per_page

    start_row = (page - 1) * rows_per_page
    end_row = start_row + rows_per_page
    page_data = df.iloc[start_row:end_row]

    table_html = page_data.to_html(classes="table table-bordered table-striped", index=False)

    return render_template(
        "combined.html",
        file_uploaded=True,
        table_html=table_html,
        filename=os.path.basename(filepath),
        page=page,
        has_prev=page > 1,
        has_next=page < total_pages,
        metrics_svm=session.get('metrics_svm'),
        metrics_ann=session.get('metrics_ann')
    )

@app.route("/delete_file", methods=["POST"])
def delete_file():
    filepath = session.get('uploaded_file')
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
        session.pop('uploaded_file', None)
        session.pop('data_shape', None)
        session.pop('current_page', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
