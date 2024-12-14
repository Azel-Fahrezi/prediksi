import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def calculate_ann(dataframe, weeks=12):
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
