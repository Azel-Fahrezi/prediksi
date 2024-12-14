import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def calculate_svm(dataframe, weeks=12):
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
