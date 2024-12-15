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

        # Predict future values
        future_index = np.arange(len(dataframe), len(dataframe) + weeks).reshape(-1, 1)
        future_index_scaled = scaler.transform(future_index)
        predictions[column] = model.predict(future_index_scaled)

        # Evaluate model
        y_pred = model.predict(X_scaled)
        metrics[column] = {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'R2': r2_score(y, y_pred)
        }

    future_dates = pd.date_range(start=dataframe.index[-1], periods=weeks + 1, freq='W')[1:]
    prediction_df = pd.DataFrame(predictions, index=future_dates)
    return prediction_df, metrics


def calculate_ann(dataframe, weeks):
    predictions = {}
    metrics = {}
    X = np.arange(len(dataframe)).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for column in dataframe.columns:
        y = dataframe[column].values

        # Create and train ANN model
        model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=500)
        model.fit(X_scaled, y)

        # Predict future values
        future_index = np.arange(len(dataframe), len(dataframe) + weeks).reshape(-1, 1)
        future_index_scaled = scaler.transform(future_index)
        predictions[column] = model.predict(future_index_scaled)

        # Evaluate model
        y_pred = model.predict(X_scaled)
        metrics[column] = {
            'MAE': mean_absolute_error(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'R2': r2_score(y, y_pred)
        }

    future_dates = pd.date_range(start=dataframe.index[-1], periods=weeks + 1, freq='W')[1:]
    prediction_df = pd.DataFrame(predictions, index=future_dates)
    return prediction_df, metrics


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

    return render_template("import.html", file_uploaded=False)


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
        predictions['index'] = predictions['index'].dt.strftime('%Y-%m-%d')  # Konversi Timestamp ke string

        # Debugging: Print isi predictions
        print(predictions.head())  # Pastikan ini menghasilkan format yang benar
        prediction_records = predictions.to_dict(orient="records")

        # Debugging: Print isi setelah to_dict
        print(prediction_records)  # Pastikan dictionary ini sesuai dengan format yang diharapkan oleh template

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
        predictions['index'] = predictions['index'].dt.strftime('%Y-%m-%d')  # Konversi Timestamp ke string

        # Debugging: Print isi predictions
        print(predictions.head())  # Pastikan ini menghasilkan format yang benar
        prediction_records = predictions.to_dict(orient="records")

        # Debugging: Print isi setelah to_dict
        print(prediction_records)  # Pastikan dictionary ini sesuai dengan format yang diharapkan oleh template

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
