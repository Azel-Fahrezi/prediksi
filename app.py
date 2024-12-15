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

def calculate_svm(dataframe, weeks):
    predictions = {}
    X = np.arange(len(dataframe)).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for column in dataframe.columns:
        y = dataframe[column].values
        model = SVR(kernel='rbf')
        model.fit(X_scaled, y)
        future_index = np.arange(len(dataframe), len(dataframe) + weeks).reshape(-1, 1)
        future_index_scaled = scaler.transform(future_index)
        predictions[column] = model.predict(future_index_scaled)

    future_dates = pd.date_range(start=dataframe.index[-1], periods=weeks + 1, freq='W')[1:]
    prediction_df = pd.DataFrame(predictions, index=future_dates)
    return prediction_df

def calculate_ann(dataframe, weeks):
    predictions = {}
    X = np.arange(len(dataframe)).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for column in dataframe.columns:
        y = dataframe[column].values
        
        model = Sequential([
            Dense(32, activation='relu', input_shape=(1,)),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        model.fit(X_scaled, y, epochs=500, verbose=0)

        future_index = np.arange(len(dataframe), len(dataframe) + weeks).reshape(-1, 1)
        future_index_scaled = scaler.transform(future_index)
        predictions[column] = model.predict(future_index_scaled).flatten()

    future_dates = pd.date_range(start=dataframe.index[-1], periods=weeks + 1, freq='W')[1:]
    prediction_df = pd.DataFrame(predictions, index=future_dates)
    return prediction_df

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Proses unggah file
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        session['uploaded_file'] = filepath

        # Membaca data CSV
        data = pd.read_csv(filepath)
        data['datum'] = pd.to_datetime(data['datum'])
        return render_template("import.html", tables=[data.to_html(classes='data', header="true", index=False)], file_uploaded=True)

    # Jika GET request
    return render_template("import.html", file_uploaded=False)

@app.route("/hasil_svm", methods=["GET", "POST"])
def hasil_svm():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))
    
    # Proses input dari form
    if request.method == "POST":
        weeks = int(request.form.get('weeks', 12))  # Ambil input dari form
        session['weeks'] = weeks  # Simpan ke session

    weeks = session.get('weeks', 12)
    data = pd.read_csv(filepath)
    data['datum'] = pd.to_datetime(data['datum'])
    data.set_index('datum', inplace=True)
    predictions = calculate_svm(data, weeks)
    return render_template("hasil_svm.html", predictions=[predictions.to_html(classes='data', header="true")])

@app.route("/hasil_ann", methods=["GET", "POST"])
def hasil_ann():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('index'))
    
    # Proses input dari form
    if request.method == "POST":
        weeks = int(request.form.get('weeks', 12))  # Ambil input dari form
        session['weeks'] = weeks  # Simpan ke session

    weeks = session.get('weeks', 12)
    data = pd.read_csv(filepath)
    data['datum'] = pd.to_datetime(data['datum'])
    data.set_index('datum', inplace=True)
    predictions = calculate_ann(data, weeks)
    return render_template("hasil_ann.html", predictions=[predictions.to_html(classes='data', header="true")])

if __name__ == "__main__":
    app.run(debug=True, port=5001)
