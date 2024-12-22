from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)
app.secret_key = 'secretkey123'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi untuk menghitung prediksi SVM
def calculate_svm(dataframe, Jumlah_Stok, Musim_Periodik, Penjualan_Sebelumnya):
    predictions = {}
    metrics = {}

    # Proses Label Encoding untuk Musim_Periodik
    encoder = LabelEncoder()
    dataframe['Musim_Periodik'] = encoder.fit_transform(dataframe['Musim_Periodik'])
    Musim_Periodik_encoded = encoder.transform([Musim_Periodik])[0]

    # Gunakan kolom selain 'datum'
    X = dataframe[['Jumlah_Stok', 'Musim_Periodik', 'Penjualan_Sebelumnya']].values
    y = dataframe['Penjualan_Prediksi'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model SVM
    model = SVR(kernel='rbf')
    model.fit(X_scaled, y)

    # Prediksi nilai di masa depan
    future_X = np.array([[Jumlah_Stok, Musim_Periodik_encoded, Penjualan_Sebelumnya]])
    future_X_scaled = scaler.transform(future_X)
    predictions['Penjualan_Prediksi'] = model.predict(future_X_scaled)

    # Evaluasi model
    y_pred = model.predict(X_scaled)
    metrics['SVM'] = {
        'MAE': mean_absolute_error(y, y_pred),
        'MSE': mean_squared_error(y, y_pred),
        'R2': r2_score(y, y_pred)
    }

    return predictions, metrics

# Fungsi untuk menghitung prediksi ANN
def calculate_ann(dataframe, Jumlah_Stok, Musim_Periodik, Penjualan_Sebelumnya):
    predictions = {}
    metrics = {}

    # Proses Label Encoding untuk Musim_Periodik
    encoder = LabelEncoder()
    # Pastikan hanya melakukan fit_transform jika data Musim_Periodik sudah ada
    if 'Musim_Periodik' not in dataframe.columns:
        raise ValueError("Kolom 'Musim_Periodik' tidak ditemukan dalam dataframe")
    
    dataframe['Musim_Periodik'] = encoder.fit_transform(dataframe['Musim_Periodik'])
    
    # Jika Musim_Periodik ada dalam kategori baru, maka handle itu
    if Musim_Periodik not in encoder.classes_:
        raise ValueError(f"Kategori '{Musim_Periodik}' tidak ditemukan dalam data pelatihan")
    
    Musim_Periodik_encoded = encoder.transform([Musim_Periodik])[0]

    # Gunakan kolom selain 'datum'
    X = dataframe[['Jumlah_Stok', 'Musim_Periodik', 'Penjualan_Sebelumnya']].values
    y = dataframe['Penjualan_Prediksi'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model ANN
    model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(X_scaled, y)

    # Prediksi nilai di masa depan
    future_X = np.array([[Jumlah_Stok, Musim_Periodik_encoded, Penjualan_Sebelumnya]])
    future_X_scaled = scaler.transform(future_X)
    predictions['Penjualan_Prediksi'] = model.predict(future_X_scaled)

    # Evaluasi model
    y_pred = model.predict(X_scaled)
    metrics['ANN'] = {
        'MAE': mean_absolute_error(y, y_pred),
        'MSE': mean_squared_error(y, y_pred),
        'R2': r2_score(y, y_pred)
    }

    return predictions, metrics

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

@app.route("/hasil_svm", methods=["POST", "GET"])
def hasil_svm():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    data = pd.read_csv(filepath)

    if request.method == "POST":
        Jumlah_Stok = float(request.form["Jumlah_Stok"])
        Musim_Periodik = request.form["Musim_Periodik"]  # Tidak diubah menjadi float
        Penjualan_Sebelumnya = float(request.form["Penjualan_Sebelumnya"])

        predictions, _ = calculate_svm(data, Jumlah_Stok, Musim_Periodik, Penjualan_Sebelumnya)

        # Kembalikan indeks sebagai kolom dan ubah Timestamp ke string
        prediction_records = [{
            'index': 'Prediksi',
            'Penjualan_Prediksi': predictions['Penjualan_Prediksi'][0]
        }]

        return render_template("hasil_svm.html", predictions=prediction_records)

    return render_template("hasil_svm.html")

@app.route("/hasil_ann", methods=["POST", "GET"])
def hasil_ann():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    data = pd.read_csv(filepath)

    if request.method == "POST":
        try:
            Jumlah_Stok = float(request.form["Jumlah_Stok"])
            Musim_Periodik = request.form["Musim_Periodik"]  # Nilai Musim_Periodik tetap string
            Penjualan_Sebelumnya = float(request.form["Penjualan_Sebelumnya"])

            predictions, _ = calculate_ann(data, Jumlah_Stok, Musim_Periodik, Penjualan_Sebelumnya)

            # Kembalikan hasil prediksi dalam bentuk record
            prediction_records = [{
                'index': 'Prediksi',
                'Penjualan_Prediksi': predictions['Penjualan_Prediksi'][0]
            }]

            return render_template("hasil_ann.html", predictions=prediction_records)

        except ValueError as e:
            # Jika ada kesalahan dalam proses, misalnya kategori tidak ditemukan
            return render_template("hasil_ann.html", error_message=str(e))

    return render_template("hasil_ann.html")

@app.route("/metrics", methods=["GET"])
def metrics():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    data = pd.read_csv(filepath)

    _, metrics_svm = calculate_svm(data, 0, 'Normal', 0)  # Update sesuai dengan data yang Anda punya
    _, metrics_ann = calculate_ann(data, 0, 'Normal', 0)  # Update sesuai dengan data yang Anda punya

    return render_template("metrics.html", metrics_svm=metrics_svm, metrics_ann=metrics_ann)

if __name__ == "__main__":
    app.run(debug=True)
