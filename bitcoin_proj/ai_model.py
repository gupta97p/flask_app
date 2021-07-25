import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import os
import keras
import requests
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler



def arima_model():
    def download_file_from_google_drive(id, destination):
            URL = "https://docs.google.com/uc?export=download"

            session = requests.Session()

            response = session.get(URL, params = { 'id' : id }, stream = True)
            token = get_confirm_token(response)

            if token:
                params = { 'id' : id, 'confirm' : token }
                response = session.get(URL, params = params, stream = True)

            save_response_content(response, destination)   

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)    
    destination = 'btc_dataset.csv'
    if os.path.isfile(destination)==False:
        file_id = '114CAwuP38I2eoJPKdsT_itGPyKzf1t8p'
        download_file_from_google_drive(file_id, destination)
    df = pd.read_csv(destination)
    df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
    group = df.groupby('date')
    Real_Price = group['Weighted_Price'].mean()

    prediction_days = 30
    df_train= Real_Price[:len(Real_Price)-prediction_days]
    df_test= Real_Price[len(Real_Price)-prediction_days:]

    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    X_train = training_set[0:len(training_set)-1]
    y_train = training_set[1:len(training_set)]
    X_train = np.reshape(X_train, (len(X_train), 1, 1))
    
    test_set = df_test.values
    X_test = test_set[:len(test_set)-1]
    y_test = test_set[1:len(test_set)-1]

    loaded = keras.models.load_model("lstm.h5")

    inputs = np.reshape(X_test, (len(X_test), 1))
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    predicted_BTC_price = loaded.predict(inputs)
    predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)

    plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()  
    plt.plot(y_test, color = 'red', label = 'Real BTC Price')
    plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')
    plt.title('BTC Price Prediction', fontsize=40)
    df_test = df_test.reset_index()
    x=df_test.index
    labels = df_test['date']
    plt.xticks(x, labels, rotation = 'vertical')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('BTC Price(USD)', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})
    plt.savefig('static/lstm_btc_pred.png')

    prediction_days = 365
    df_train= Real_Price[:len(Real_Price)-prediction_days]
    df_test= Real_Price[len(Real_Price)-prediction_days:]

    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    X_train = training_set[0:len(training_set)-1]
    y_train = training_set[1:len(training_set)]
    X_train = np.reshape(X_train, (len(X_train), 1, 1))

    test_set = df_test.values
    X_test = test_set[:len(test_set)-1]
    y_test = test_set[1:len(test_set)-1]

    loaded_year = keras.models.load_model("lstm_year.h5")

    inputs = np.reshape(X_test, (len(X_test), 1))
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    predicted_BTC_price = loaded_year.predict(inputs)
    predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)

    plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()  
    plt.plot(y_test, color = 'red', label = 'Real BTC Price')
    plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')
    plt.title('BTC Price Prediction', fontsize=40)


    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('BTC Price(USD)', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})
    plt.savefig('static/lstm_btc_pred_year.png')
    predicted_BTC_price = predicted_BTC_price[:len(predicted_BTC_price)-1]
    accuracy = r2_score(y_test, predicted_BTC_price)
    return accuracy
arima_model()
print("job done")