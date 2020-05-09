#LSTM stock prediciton
import math
import numpy as np; np.random.seed(1337) # for reproducibility
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #silence error messages
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from datetime import datetime, timedelta, date
import pandas as pd
import csv
import smtplib

#EMAIL
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import smtplib
import sys

stock_name = "KMD.NZ"
period = "12y"
look_back = 60
days_to_predict = 30


def fetch_data(period_to_download):
    stock = yf.Ticker(stock_name)

    df = stock.history(period=period_to_download)
    df.dropna(inplace=True)

    close_data = df.filter(["Close"])
    dataset = close_data.values
    print("length of fetched data {} ".format(len(dataset)))

    training_dataset_len = math.ceil( len(dataset) * 0.8) #train with 80% of the data
    print("length of training data {} ".format(training_dataset_len))
    return close_data, dataset, training_dataset_len,df

def scale_data(dataset):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data,scaler

def process_split_data(scaled_data,training_dataset_len,look_back):
    #create scaled x and y training datasets
    train_data = scaled_data[0:training_dataset_len, :] #return all the columns
    #split data into x and y training
    x_train = [] #independent
    y_train = [] #dependent

    for i in range(look_back, len(train_data)): #doing it in look_back day segments
        x_train.append(train_data[i-look_back:i, 0]) #0-59  (previous data )
        y_train.append(train_data[i, 0]) #look_back (value to predict)
    #convert x_train and y_train to numpy arrays

    x_train, y_train = np.array(x_train), np.array(y_train)

    #reshape the data for the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1)) #number of samples, number of time steps and number of features

    #x_train, y_train = np.nan_to_num(x_train), np.nan_to_num(y_train)

    return x_train, y_train, train_data

def neural_network(x_train, y_train):
    #LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,input_shape = (x_train.shape[1],1))) #input_shape = number of timestamps and number of features
    model.add(LSTM(50,return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience=4)

    model.compile(optimizer="adam", loss="mean_squared_error") #used to measure how well the model predicted

    model.fit(x_train, y_train, batch_size=32, epochs=1000,callbacks=[es])

    return model

def split_test_data(scaled_data,training_dataset_len,look_back,dataset):
    #create array for scaled testing dataset (20% of data)
    test_data = scaled_data[training_dataset_len - look_back: , :]

    #create x_test and y_test
    x_test = []
    y_test = dataset[training_dataset_len: , :] #all of the values we want to model to predict
    for i in range(look_back, len(test_data)):
        x_test.append(test_data[i-look_back:i, 0])

    #convert testing data to numpy arrays
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1)) #number of samples, number of timestamps, number of features( close price)

    return y_test, x_test

def backtest(model, x_test,scaler):
    #Get models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions) #unscale

    return predictions

def predict(number_of_days_to_predict, model,scaler,look_back,dataset):
    """predicts future close prices, takes the last 60 day values + number of day that you want to predict and runs them through the network"""
    prediction_list = dataset[-look_back:]
    prediction_list = np.array(prediction_list)

    for j in range(number_of_days_to_predict):
        x_data_prediction = prediction_list[-look_back:] #-60 --> 0
        x_data_prediction = x_data_prediction.reshape(look_back,1) #changes from 1d to 2d

        x_data_prediction = scaler.fit_transform(x_data_prediction)#needs a 2d array
        x_data_prediction = x_data_prediction.reshape((1, look_back, 1))

        out = model.predict(x_data_prediction)
        out = scaler.inverse_transform(out) #unscale

        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]

    return prediction_list

def predict_dates(number_of_days_to_predict,df):
    last_date = df.index[-1]
    prediction_dates = []
    prediction_dates.append(last_date)
    for number_of_days_in_future in range(number_of_days_to_predict):
        possible_future_date = last_date + timedelta(days=number_of_days_in_future+1)
        day_of_the_week = possible_future_date.isoweekday()
        if day_of_the_week <= 5: #not a weekend
            prediction_dates.append(possible_future_date)
        elif day_of_the_week == 6: #saturday
            prediction_dates.append(possible_future_date + timedelta(days=2))
        elif day_of_the_week == 7: #sunday
            prediction_dates.append(possible_future_date + timedelta(days=1))
    return prediction_dates

def predict_future_wrapper(days_to_predict,model,scaler,dataset,look_back,full_dataset):
    dataset = dataset.reshape((-1))
    forecast = predict(days_to_predict, model, scaler, look_back, dataset)
    forecast_dates = predict_dates(days_to_predict,full_dataset)

    return forecast, forecast_dates

def pretty_print(forecast,forecast_dates):
    formatted_data = {"Predictions":forecast}
    df = pd.DataFrame(formatted_data,columns=["Predictions"], index=forecast_dates)
    df.index.name = "Dates"
    print(df)

    return df

def plot_closing_prices(close_data,training_dataset_len,predictions,stock_name,days_to_predict,forecast_dates,forecast):
    train = close_data[:training_dataset_len]
    valid = close_data[training_dataset_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16,8))
    plt.title('{0} predicted closing price for the next {1} days'.format(stock_name,days_to_predict))
    plt.xlabel("Date")
    plt.ylabel("Close Price NZD ($)")
    plt.plot(train['Close']) #contains x and y data
    plt.plot(valid[['Close', 'Predictions']]) #plotting two lines both containing x and y
    plt.plot(forecast_dates,forecast)
    plt.legend(['Training Data', 'Actual Price','Past Predictions', 'Future Predictions'],loc="upper left")
    #plt.show()
    plt.savefig("{0} predicted closing price for the next {1} days.png".format(stock_name,days_to_predict))


def write_csv(stock_name, days_to_predict, forecast, forecast_dates):
    with open('{0} {1} {2} days.csv'.format(datetime.date(datetime.now()),stock_name,days_to_predict), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(forecast_dates, forecast))

def main():
    close_data, dataset, training_dataset_len, full_dataset = fetch_data(period)
    scaled_data, scaler = scale_data(dataset)
    x_train, y_train, train_data = process_split_data(scaled_data, training_dataset_len, look_back)
    model = neural_network(x_train, y_train)
    y_test, x_test = split_test_data(scaled_data, training_dataset_len, look_back, dataset)
    past_predictions = backtest(model, x_test,scaler)
    forecast, forecast_dates = predict_future_wrapper(days_to_predict, model, scaler, dataset, look_back, full_dataset)
    plot_closing_prices(close_data, training_dataset_len, past_predictions, stock_name, days_to_predict, forecast_dates, forecast)
    write_csv(stock_name, days_to_predict, forecast, forecast_dates)
    predictions_dataframe = pretty_print(forecast,forecast_dates)
    plt.show()

main()
