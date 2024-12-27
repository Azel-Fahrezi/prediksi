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

    # Validasi apakah kolom yang diperlukan ada
    required_columns = ['Jumlah_Stok', 'Musim_Periodik', 'Penjualan_Sebelumnya', 'Penjualan_Prediksi']
    for col in required_columns:
        if col not in dataframe.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataframe")

    # Label Encoding untuk Musim_Periodik
    encoder = LabelEncoder()
    dataframe['Musim_Periodik'] = encoder.fit_transform(dataframe['Musim_Periodik'])

    # Tambahkan kategori baru jika Musim_Periodik input tidak ada di data pelatihan
    if Musim_Periodik not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, Musim_Periodik)

    Musim_Periodik_encoded = encoder.transform([Musim_Periodik])[0]

    # Gunakan kolom untuk fitur dan target
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

            # Membaca data CSV
            df = pd.read_csv(filepath)
            
            # Validasi kolom yang diperlukan
            required_columns = ['Jumlah_Stok', 'Musim_Periodik', 'Penjualan_Sebelumnya', 'Penjualan_Prediksi']
            for col in required_columns:
                if col not in df.columns:
                    return f"Kolom '{col}' tidak ditemukan dalam file yang diunggah. Pastikan file memiliki kolom: {', '.join(required_columns)}."

            # Simpan ke session
            session['uploaded_file'] = filepath

            # Konversi data CSV ke dictionary
            uploaded_data = df.to_dict(orient='records')
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

@app.route("/testing", methods=["POST", "GET"])
def testing():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))

    data = pd.read_csv(filepath)

    predictions_svm = None
    predictions_ann = None

    if request.method == "POST":
        try:
            Jumlah_Stok = float(request.form["Jumlah_Stok"])
            Musim_Periodik = request.form["Musim_Periodik"]
            Penjualan_Sebelumnya = float(request.form["Penjualan_Sebelumnya"])

            # Hitung prediksi untuk SVM
            predictions_svm, _ = calculate_svm(data, Jumlah_Stok, Musim_Periodik, Penjualan_Sebelumnya)

            # Hitung prediksi untuk ANN
            predictions_ann, _ = calculate_ann(data, Jumlah_Stok, Musim_Periodik, Penjualan_Sebelumnya)

            return render_template("testing.html", 
                                   predictions_svm=predictions_svm, 
                                   predictions_ann=predictions_ann)

        except ValueError as e:
            return render_template("testing.html", error_message=str(e))

    return render_template("testing.html")

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
