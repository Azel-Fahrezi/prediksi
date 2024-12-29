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
            return redirect(url_for('training'))
        except Exception as e:
            os.remove(filepath)
            return f"Error membaca file: {str(e)}"

    return render_template("import.html")

@app.route("/training", methods=["GET", "POST"])
def training():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    data = pd.read_csv(filepath)
    target_accuracy = 95.0
    max_iter = 500

    svm_model, svm_scaler, svm_label_mapping, metrics_svm, svm_labels = train_model(data, "SVM", max_iter, target_accuracy)
    ann_model, ann_scaler, ann_label_mapping, metrics_ann, ann_labels = train_model(data, "ANN", max_iter, target_accuracy)

    # Simpan model dan skalar ke file
    svm_model_file = os.path.join(SAVE_FOLDER, 'svm_model.pkl')
    ann_model_file = os.path.join(SAVE_FOLDER, 'ann_model.pkl')
    svm_scaler_file = os.path.join(SAVE_FOLDER, 'svm_scaler.pkl')
    ann_scaler_file = os.path.join(SAVE_FOLDER, 'ann_scaler.pkl')

    with open(svm_model_file, 'wb') as f:
        pickle.dump(svm_model, f)
    with open(ann_model_file, 'wb') as f:
        pickle.dump(ann_model, f)
    with open(svm_scaler_file, 'wb') as f:
        pickle.dump(svm_scaler, f)
    with open(ann_scaler_file, 'wb') as f:
        pickle.dump(ann_scaler, f)

    # Simpan nama file ke session
    session['svm_model_file'] = 'svm_model.pkl'
    session['ann_model_file'] = 'ann_model.pkl'
    session['svm_scaler_file'] = 'svm_scaler.pkl'
    session['ann_scaler_file'] = 'ann_scaler.pkl'
    session['svm_label_mapping'] = svm_label_mapping
    session['ann_label_mapping'] = ann_label_mapping
    session['original_labels'] = svm_labels  # Simpan label asli
    print(f"Original labels disimpan: {svm_labels}")

    return render_template("training.html", metrics_svm=metrics_svm["SVM"], metrics_ann=metrics_ann["ANN"])

@app.route("/testing", methods=["POST", "GET"])
def testing():
    if 'svm_model_file' not in session or 'ann_model_file' not in session:
        return redirect(url_for('training'))

    # Muat model dan skalar dari file
    svm_model_file = os.path.join(SAVE_FOLDER, session['svm_model_file'])
    ann_model_file = os.path.join(SAVE_FOLDER, session['ann_model_file'])
    svm_scaler_file = os.path.join(SAVE_FOLDER, session['svm_scaler_file'])
    ann_scaler_file = os.path.join(SAVE_FOLDER, session['ann_scaler_file'])

    with open(svm_model_file, 'rb') as f:
        svm_model = pickle.load(f)
    with open(ann_model_file, 'rb') as f:
        ann_model = pickle.load(f)
    with open(svm_scaler_file, 'rb') as f:
        svm_scaler = pickle.load(f)
    with open(ann_scaler_file, 'rb') as f:
        ann_scaler = pickle.load(f)

    svm_label_mapping = session['svm_label_mapping']
    ann_label_mapping = session['ann_label_mapping']
    original_labels = session.get('original_labels', [])
    print(f"Original labels di testing: {original_labels}")

    if not original_labels:
        return render_template("testing.html", error_message="Tidak ada label yang valid ditemukan. Silakan lakukan pelatihan ulang.", original_labels=original_labels)

    if request.method == "POST":
        try:
            features = {
                "Jumlah_Stok": float(request.form["Jumlah_Stok"]),
                "Musim_Periodik": request.form["Musim_Periodik"],
                "Penjualan_Sebelumnya": float(request.form["Penjualan_Sebelumnya"]),
            }
            print(f"Features input: {features}")
            prediction_svm = predict_model(svm_model, svm_scaler, svm_label_mapping, features)
            prediction_ann = predict_model(ann_model, ann_scaler, ann_label_mapping, features)
            return render_template("testing.html", predictions_svm={"Penjualan_Prediksi": prediction_svm}, predictions_ann={"Penjualan_Prediksi": prediction_ann}, original_labels=original_labels)
        except ValueError as e:
            return render_template("testing.html", error_message=str(e), original_labels=original_labels)

    return render_template("testing.html", original_labels=original_labels)

@app.route("/metrics", methods=["GET"])
def metrics():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    data = pd.read_csv(filepath)
    target_accuracy = 95.0
    max_iter = 500

    _, _, _, metrics_svm, _ = train_model(data, "SVM", max_iter, target_accuracy)
    _, _, _, metrics_ann, _ = train_model(data, "ANN", max_iter, target_accuracy)

    return render_template("metrics.html", metrics_svm=metrics_svm["SVM"], metrics_ann=metrics_ann["ANN"])

if __name__ == "__main__":
    app.run(debug=True)
