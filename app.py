from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)
app.secret_key = 'secretkey123'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi untuk menghitung prediksi SVM
def calculate_svm(dataframe, weeks):
    predictions = {}
    metrics = {}
    X = np.arange(len(dataframe)).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for column in dataframe.columns:
        y = dataframe[column].values
        model = SVR(kernel='rbf')
        model.fit(X_scaled, y)

        # Prediksi nilai di masa depan
        future_index = np.arange(len(dataframe), len(dataframe) + weeks).reshape(-1, 1)
        future_index_scaled = scaler.transform(future_index)
        predictions[column] = model.predict(future_index_scaled)

        # Evaluasi model
        y_pred = model.predict(X_scaled)
        metrics[column] = {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'R2': r2_score(y, y_pred)
        }

    future_dates = pd.date_range(start=dataframe.index[-1], periods=weeks + 1, freq='W')[1:]
    prediction_df = pd.DataFrame(predictions, index=future_dates)
    return prediction_df, metrics

# Fungsi untuk menghitung prediksi ANN
def calculate_ann(dataframe, weeks):
    predictions = {}
    metrics = {}
    X = np.arange(len(dataframe)).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for column in dataframe.columns:
        y = dataframe[column].values

        # Membuat dan melatih model ANN
        model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=500)
        model.fit(X_scaled, y)

        # Prediksi nilai di masa depan
        future_index = np.arange(len(dataframe), len(dataframe) + weeks).reshape(-1, 1)
        future_index_scaled = scaler.transform(future_index)
        predictions[column] = model.predict(future_index_scaled)

        # Evaluasi model
        y_pred = model.predict(X_scaled)
        metrics[column] = {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'R2': r2_score(y, y_pred)
        }

    future_dates = pd.date_range(start=dataframe.index[-1], periods=weeks + 1, freq='W')[1:]
    prediction_df = pd.DataFrame(predictions, index=future_dates)
    return prediction_df, metrics

# Variabel global untuk menyimpan data
uploaded_data = []
filename = ""

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_data, filename

    # Periksa apakah file sudah ada di session
    if 'uploaded_file' in session:
        uploaded_data = pd.read_csv(session['uploaded_file']).to_dict(orient='records')
        filename = session.get('uploaded_file').split('/')[-1]
        return redirect(url_for('paginate', page=1))

    if request.method == "POST":
        action = request.form.get('action')

        if action == "upload":
            if 'file' not in request.files:
                return "No file part"
            file = request.files['file']
            if file.filename == '':
                return "No file selected"

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Simpan ke session
            session['uploaded_file'] = filepath

            # Membaca data CSV dan mengonversinya ke dictionary
            uploaded_data = pd.read_csv(filepath).to_dict(orient='records')
            filename = file.filename

            return redirect(url_for('paginate', page=1))

        elif action == "delete":
            uploaded_data = []
            filename = ""
            session.pop('uploaded_file', None)  # Hapus data session
            return render_template("import.html", file_uploaded=False)

    return render_template("import.html", file_uploaded=False)

@app.route("/page/<int:page>")
def paginate(page):
    global uploaded_data

    # Jika data kosong, redirect ke halaman utama
    if not uploaded_data:
        return redirect(url_for('index'))

    # Hitung batas paginasi
    rows_per_page = 10
    start = (page - 1) * rows_per_page
    end = start + rows_per_page
    page_data = uploaded_data[start:end]

    # Konversi halaman data ke HTML
    df = pd.DataFrame(page_data)
    table_html = df.to_html(classes='table table-striped table-bordered table-hover', index=False, border=0)

    # Hitung apakah ada halaman sebelumnya atau berikutnya
    has_next = end < len(uploaded_data)
    has_prev = start > 0

    return render_template(
        "import.html",
        table_html=table_html,
        file_uploaded=True,
        filename=filename,
        page=page,
        has_next=has_next,
        has_prev=has_prev,
    )

@app.route("/delete", methods=["POST"])
def delete():
    filename = request.form.get('filename')
    if filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)  # Hapus file
            print(f"File {filename} berhasil dihapus")
    return redirect(url_for('index'))

@app.route("/hasil_svm", methods=["POST", "GET"])
def hasil_svm():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    data = pd.read_csv(filepath)
    data['datum'] = pd.to_datetime(data['datum'])
    data.set_index('datum', inplace=True)

    if request.method == "POST":
        weeks = int(request.form["weeks"])
        predictions, _ = calculate_svm(data, weeks)

        # Kembalikan indeks sebagai kolom dan ubah Timestamp ke string
        predictions.reset_index(inplace=True)
        predictions['index'] = predictions['index'].dt.strftime('%Y-%m-%d')

        # Debugging: Print isi predictions
        print(predictions.head())

        prediction_records = predictions.to_dict(orient="records")

        # Debugging: Print isi setelah to_dict
        print(prediction_records)

        return render_template("hasil_svm.html", predictions=prediction_records)

    return render_template("hasil_svm.html")

@app.route("/hasil_ann", methods=["POST", "GET"])
def hasil_ann():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    data = pd.read_csv(filepath)
    data['datum'] = pd.to_datetime(data['datum'])
    data.set_index('datum', inplace=True)

    if request.method == "POST":
        weeks = int(request.form["weeks"])
        predictions, _ = calculate_ann(data, weeks)

        # Kembalikan indeks sebagai kolom dan ubah Timestamp ke string
        predictions.reset_index(inplace=True)
        predictions['index'] = predictions['index'].dt.strftime('%Y-%m-%d')

        # Debugging: Print isi predictions
        print(predictions.head())

        prediction_records = predictions.to_dict(orient="records")

        # Debugging: Print isi setelah to_dict
        print(prediction_records)

        return render_template("hasil_ann.html", predictions=prediction_records)

    return render_template("hasil_ann.html")

@app.route("/metrics", methods=["GET"])
def metrics():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    data = pd.read_csv(filepath)
    data['datum'] = pd.to_datetime(data['datum'])
    data.set_index('datum', inplace=True)

    weeks_svm = session.get('weeks_svm', 12)
    weeks_ann = session.get('weeks_ann', 12)

    _, metrics_svm = calculate_svm(data, weeks_svm)
    _, metrics_ann = calculate_ann(data, weeks_ann)

    return render_template("metrics.html", metrics_svm=metrics_svm, metrics_ann=metrics_ann)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
