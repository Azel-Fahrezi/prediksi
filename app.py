from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)
app.secret_key = 'secretkey123'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi Perhitungan SVM
def calculate_svm(dataframe):
    predictions = {}
    X = np.arange(len(dataframe)).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for column in dataframe.columns:
        y = dataframe[column].values
        model = SVR(kernel='rbf')
        model.fit(X_scaled, y)
        future_index = np.arange(len(dataframe), len(dataframe) + 12).reshape(-1, 1)
        future_index_scaled = scaler.transform(future_index)
        predictions[column] = model.predict(future_index_scaled)

    future_dates = pd.date_range(start=dataframe.index[-1], periods=13, freq='W')[1:]
    prediction_df = pd.DataFrame(predictions, index=future_dates)
    return prediction_df

# Fungsi Perhitungan ANN
def calculate_ann(dataframe):
    predictions = {}
    X = np.arange(len(dataframe)).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for column in dataframe.columns:
        y = dataframe[column].values
        
        # Membuat Model ANN
        model = Sequential([
            Dense(32, activation='relu', input_shape=(1,)),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        
        # Melatih model
        model.fit(X_scaled, y, epochs=500, verbose=0)
        
        # Prediksi 12 minggu ke depan
        future_index = np.arange(len(dataframe), len(dataframe) + 12).reshape(-1, 1)
        future_index_scaled = scaler.transform(future_index)
        predictions[column] = model.predict(future_index_scaled).flatten()

    future_dates = pd.date_range(start=dataframe.index[-1], periods=13, freq='W')[1:]
    prediction_df = pd.DataFrame(predictions, index=future_dates)
    return prediction_df

@app.route("/")
def index():
    return render_template("import.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    session['uploaded_file'] = filepath
    return redirect(url_for('show_data'))

@app.route("/data")
def show_data():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))
    data = pd.read_csv(filepath)
    data['datum'] = pd.to_datetime(data['datum'])
    # Tampilkan 'datum' sebagai kolom, bukan indeks
    return render_template("data.html", tables=[data.to_html(classes='data', header="true", index=False)])

@app.route("/hasil_svm")
def hasil_svm():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))
    data = pd.read_csv(filepath)
    data['datum'] = pd.to_datetime(data['datum'])
    data.set_index('datum', inplace=True)
    predictions = calculate_svm(data)
    return render_template("hasil_svm.html", predictions=[predictions.to_html(classes='data', header="true")])

@app.route("/hasil_ann")
def hasil_ann():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))
    data = pd.read_csv(filepath)
    data['datum'] = pd.to_datetime(data['datum'])
    data.set_index('datum', inplace=True)
    predictions = calculate_ann(data)
    return render_template("hasil_ann.html", predictions=[predictions.to_html(classes='data', header="true")])

if __name__ == "__main__":
    app.run(debug=True)
