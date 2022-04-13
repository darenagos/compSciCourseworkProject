import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import pandas_datareader as pdr
import datetime as dt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from datetime import date, datetime, time, timedelta, timezone

import mplfinance as mpf
import tkinter as tk
from tkinter import *

from tkinter import messagebox #tkinter module for pop up messages

# gui uses tkinter library

from tkinter import *

# global variables

CANDLESTICK_CHART = "CANDLESTICK_CHART"
LINE_CHART = "LINE_CHART"

global currentChart
global canvas



currentChart = "not initialised"

root = Tk()

root.title('Stock Program')
root.geometry("1200x800")

toolbarFrame = Frame(master=root)
toolbarFrame.place(relx=0.5, rely=0.65, anchor=CENTER)


def menuOptGoog():
    labelTitle = Label(root, text="Google", font=('Helvetica', 20))
    labelTitle.place(relx=0.5, y=10, anchor=CENTER)

    # CHART BUTTONS

    googleLineChart = Button(root, text="Line Chart", command=GOOG_stock_line, height=1, width=50)
    googleLineChart.place(relx=0.2, y=40, anchor=CENTER)

    googleCandleChart = Button(root, text="Candlestick Chart", command=GOOG_stock_candlestick, height=1, width=40)
    googleCandleChart.place(relx=0.5, y=40, anchor=CENTER)

    googlePredictionLine = Button(root, text="Prediction Line", command=predictionTool, height=1, width=50)
    googlePredictionLine.place(relx=0.8, y=40, anchor=CENTER)

    # SELECT TIME INTERVAL BUTTONS

    googDay = Button(root, text="1 day", command=oneDay, height=1, width=20)
    googDay.place(relx=0.2, rely=0.7, anchor=CENTER)
    googWeek = Button(root, text="1 week", command=oneWeek, height=1, width=20)
    googWeek.place(relx=0.4, rely=0.7, anchor=CENTER)
    googMonth = Button(root, text="1 month", command=oneMonth, height=1, width=20)
    googMonth.place(relx=0.6, rely=0.7, anchor=CENTER)
    googYear = Button(root, text="1 year", command=oneYear, height=1, width=20)
    googYear.place(relx=0.8, rely=0.7, anchor=CENTER)

    var = tk.StringVar(root, ' ')
    l = tk.Label(root, textvariable=var, bg='light grey', font=('Arial', 12), width=54, height=20)
    l.place(relx=0.5, rely=0.35, anchor=CENTER)

    # PREDICTION TEXT

    labelNextDayPrediction = Label(root, text="Predicted closing price for the following day:", font=('Helvetica', 10))
    labelNextDayPrediction.place(relx=0.5, rely=0.8, anchor=E)
    var = tk.StringVar(root, ' ')
    l = tk.Label(root, textvariable=var, bg='light grey', font=('Arial', 12), width=15, height=2)
    l.place(relx=0.5, rely=0.8, anchor=W)

    # #defining toolbar frame
    # toolbarFrame = Frame(master=root)
    # toolbarFrame.place(relx=0.5, rely=0.65, anchor=CENTER)

def errorMessage():
    messagebox.showerror('User action required:', 'Please select type of graph to be displayed for your chosen time interval!')

def clearToolbar():
    for widgets in toolbarFrame.winfo_children():
     widgets.destroy()


#global toolbarFrame
#def clearToolbar():

def oneDay():
   # canvas.delete("All")
   # clearToolbar()

    print("In oneDay(). currentChart =", currentChart) #for me to see in the sys out
    if currentChart == LINE_CHART:

        return googOneDayLineChart()

    elif currentChart == CANDLESTICK_CHART:
        return googOneDayCandleStickChart()
    else:
        return errorMessage()
    #toolbarFrame.destroy()



def oneWeek():
   # clearToolbar()
   # canvas.delete("All")
    print("In oneWeek(). currentChart =", currentChart)
    if currentChart == LINE_CHART:

        return googOneWeekLineChart()

    elif currentChart == CANDLESTICK_CHART:
        return googOneWeekCandleStickChart()
    else:
        return errorMessage()


def oneMonth():
    print("In oneMonth(). currentChart =", currentChart)
    if currentChart == LINE_CHART:

        return googOneMonthLineChart()

    elif currentChart == CANDLESTICK_CHART:
        return googOneMonthCandleStickChart()
    else:
        return errorMessage()


def oneYear():
    print("In oneYear(). currentChart =", currentChart)
    if currentChart == LINE_CHART:

        return googOneYearLineChart()

    elif currentChart == CANDLESTICK_CHART:
        return googOneYearCandleStickChart()
    else:
        return errorMessage()


# def _clear():
#     for item in canvas.get_tk_widget().find_all():
#        canvas.get_tk_widget().delete(item)

# def clear_frame():
#    for widgets in toolbarFrame.winfo_children():
#       widgets.destroy()




def googOneDayLineChart():

    plt.close()
    stockOne = ['GOOG']
    today = date.today()
    yesterday = today - timedelta(days=2)
    dataOne = pdr.get_data_yahoo(stockOne, start=yesterday)['Close']
    dataOne.head()

    plt.grid(True, color='k', linestyle=':')
    rcParams['figure.figsize'] = 12, 6
    plt.plot(dataOne.GOOG)
    plt.grid(True, color='k', linestyle=':')
    plt.title("GOOG Prices")
    plt.ylabel("Price")

    fig = Figure(figsize=(5, 4), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=45)
    plot1.plot(dataOne)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER) #35

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)



    # toolbarFrame = Frame(master=root)
    # toolbarFrame.place(relx=0.5, rely=0.65, anchor=CENTER)
    # toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
    #toolbar = NavigationToolbar2Tk(frame, root)
    canvas.draw()
    #toolbar.destroy()
    # for widgets in toolbarFrame.winfo_children():
    #     widgets.destroy()



    plt.close()

def googOneWeekLineChart():
    stockOne = ['GOOG']
    today = date.today()
    yesterday = today - timedelta(days=7)
    dataOne = pdr.get_data_yahoo(stockOne, start=yesterday)['Close']
    dataOne.head()

    plt.grid(True, color='k', linestyle=':')
    rcParams['figure.figsize'] = 12, 6
    plt.plot(dataOne.GOOG)

    plt.grid(True, color='k', linestyle=':')
    plt.title("GOOG Prices")
    plt.xlabel("Date")


    fig = Figure(figsize=(5, 4), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=45)
    plot1.plot(dataOne)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER) #0.5
    mpl.rcParams['toolbar'] = 'None'

    clearToolbar()

    # for widgets in toolbarFrame.winfo_children():
    #  widgets.destroy()

    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)



    # toolbar = NavigationToolbar2Tk(canvas, root)
    # toolbar.place(relx=0.5, rely=0.65, anchor=CENTER)
    canvas.draw()
    #toolbar.destroy()


    #plt.close()


def googOneMonthLineChart():
    #plt.close()
    stockOne = ['GOOG']
    today = date.today()
    yesterday = today - timedelta(days=30)
    dataOne = pdr.get_data_yahoo(stockOne, start=yesterday)['Close']
    dataOne.head()

    plt.grid(True, color='k', linestyle=':')
    rcParams['figure.figsize'] = 12, 6
    plt.plot(dataOne.GOOG)
    plt.grid(True, color='k', linestyle=':')
    plt.title("GOOG Prices")
    plt.ylabel("Price")

    fig = Figure(figsize=(5, 4), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=45)
    plot1.plot(dataOne)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER)

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)



    # toolbar = NavigationToolbar2Tk(canvas, root)
    # toolbar.place(relx=0.5, rely=0.65, anchor=CENTER)
    canvas.draw()
    #plt.close()


def googOneYearLineChart():
   # plt.close()
    stockOne = ['GOOG']
    today = date.today()
    yesterday = today - timedelta(days=365)
    dataOne = pdr.get_data_yahoo(stockOne, start=yesterday)['Close']
    dataOne.head()

    plt.grid(True, color='k', linestyle=':')
    rcParams['figure.figsize'] = 12, 6
    plt.plot(dataOne.GOOG)
    plt.grid(True, color='k', linestyle=':')
    plt.title("GOOG Prices")
    plt.ylabel("Price")

    fig = Figure(figsize=(5, 4), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=45)
    plot1.plot(dataOne)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER)

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)



    # toolbar = NavigationToolbar2Tk(canvas, root)
    # toolbar.place(relx=0.5, rely=0.65, anchor=CENTER)
    canvas.draw()
   # plt.close()


def GOOG_stock_line():

    #plt.close()
    stockOne = ['GOOG']
    dataOne = pdr.get_data_yahoo(stockOne, start='2019-01-01')['Close']
    dataOne.head()

    rcParams['figure.figsize'] = 12, 6
    plt.plot(dataOne.GOOG)
    plt.grid(True, color='k', linestyle=':')
    plt.title("GOOG Prices")
    plt.ylabel("Price")

    fig = Figure(figsize=(5, 4), dpi=100)
    plot1 = fig.add_subplot(111)
    fig.autofmt_xdate(rotation=45)
    plot1.plot(dataOne)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER)

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
    #frame.place(relx=0.5, rely=0.65, anchor=CENTER)
    #toolbar.place(relx=0.5, rely=0.65, anchor=CENTER)
    canvas.draw()
    global currentChart
    currentChart = LINE_CHART
    #here the program defines the glabal variable currentChart as line chart
    #when the user selects the time intervals the program will check to see if the
    #current chart = line chart where it will then execute the time interval for a line chart
    #plt.close()
    #toolbarFrame.destroy()
    # for widgets in toolbarFrame.winfo_children():
    #     widgets.destroy()

def GOOG_stock_candlestick():
    plt.close()
    start = dt.datetime(2019, 1, 1)

    end = dt.datetime.now()

    data = web.DataReader("GOOG", 'yahoo', start, end)

    colors = mpf.make_marketcolors(up="green", down="red", wick="inherit", edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style='binance', marketcolors=colors)
    fig, axlist = mpf.plot(data, type="candle", style=mpf_style, volume=True, figsize=(5, 4), returnfig=True, datetime_format='%b %d')

    canvas = FigureCanvasTkAgg(fig)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER)

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)


    #plt.close()
    global currentChart
    currentChart = CANDLESTICK_CHART


def googOneDayCandleStickChart():
    plt.close()
    today = date.today()
    yesterday = today - timedelta(days=2)

    data = web.DataReader("GOOG", 'yahoo', start=yesterday)

    colors = mpf.make_marketcolors(up="green", down="red", wick="inherit", edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style='binance', marketcolors=colors)
    fig, axlist = mpf.plot(data, type="candle", style=mpf_style, volume=True, figsize=(5, 4), returnfig=True)

    canvas = FigureCanvasTkAgg(fig)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER)

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

    plt.close()


def googOneWeekCandleStickChart():
    plt.close()
    today = date.today()
    yesterday = today - timedelta(days=7)

    data = web.DataReader("GOOG", 'yahoo', start=yesterday)

    colors = mpf.make_marketcolors(up="green", down="red", wick="inherit", edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style='binance', marketcolors=colors)
    fig, axlist = mpf.plot(data, type="candle", style=mpf_style, volume=True, figsize=(5, 4), returnfig=True)

    canvas = FigureCanvasTkAgg(fig)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER)

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

    plt.close()


def googOneMonthCandleStickChart():
    plt.close()
    today = date.today()
    yesterday = today - timedelta(days=30)

    data = web.DataReader("GOOG", 'yahoo', start=yesterday)

    colors = mpf.make_marketcolors(up="green", down="red", wick="inherit", edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style='binance', marketcolors=colors)
    fig, axlist = mpf.plot(data, type="candle", style=mpf_style, volume=True, figsize=(5, 4), returnfig=True)

    canvas = FigureCanvasTkAgg(fig)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER)

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

    plt.close()


def googOneYearCandleStickChart():
    plt.close()
    today = date.today()
    yesterday = today - timedelta(days=365)

    data = web.DataReader("GOOG", 'yahoo', start=yesterday)

    colors = mpf.make_marketcolors(up="green", down="red", wick="inherit", edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style='binance', marketcolors=colors)
    fig, axlist = mpf.plot(data, type="candle", style=mpf_style, volume=True, figsize=(5, 4), returnfig=True, datetime_format='%b %d')

    canvas = FigureCanvasTkAgg(fig)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER)

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

    plt.close()


# def TSLA_stock():
#     plt.close()
#     stockOne = ['TSLA']
#     dataOne = pdr.get_data_yahoo(stockOne, start='2019-01-01')['Close']
#     dataOne.head()
#
#     rcParams['figure.figsize'] = 12, 6
#     plt.plot(dataOne.TSLA)
#     plt.grid(True, color='k', linestyle=':')
#     plt.title("TSLA Prices")
#     plt.xlabel("Date")
#
#     fig = Figure(figsize=(5, 4), dpi=100)
#     plot1 = fig.add_subplot(111)
#     plot1.plot(dataOne)
#
#     canvas = FigureCanvasTkAgg(fig, master=root)
#     canvas.get_tk_widget().place(x=380, y=100)
#     toolbar = NavigationToolbar2Tk(canvas, root)
#     toolbar.place(relx=0.5, rely=0.65, anchor=CENTER)
#     canvas.draw()
#
#
# def menuOptTsla():
#     labelTitle = Label(root, text="Tesla", font=('Helvetica', 10))
#     labelTitle.place(x=600, y=0)
#
#     tslaLineChart = Button(root, text="Line Chart", command=TSLA_stock, padx=20, pady=5)
#     tslaLineChart.place(x=400, y=30)
#
#     tslaCandleChart = Button(root, text="Candlestick Chart", command=None, padx=10, pady=5)
#     tslaCandleChart.place(x=560, y=30)
#
#     tslaPredictionLine = Button(root, text="Prediction Line", command=None, padx=10, pady=5)
#     tslaPredictionLine.place(x=720, y=30)


#####################  PREDICTION LINE

company = 'GOOG'


def predictionTool():
    start = dt.datetime(2019, 1, 1) #21
    end = dt.datetime(2022, 1, 1)

    data = web.DataReader(company, 'yahoo', start, end)

    print(data)

    # prepare data - scale down all the data values recieved so that they fit between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))  # (min, max), default=(0, 1)

    # only going to transoform closing price as i am predicting only closing price

    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60  # how many days to look inot past to base prediction on for next day

    # define two empty lists

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):  # up until the scaled data from 60th index till last index
        x_train.append(scaled_data[x - prediction_days:x,0])  # add value to x train wtih each iteration - will use 60
        # values and then the next value so that model can learn to predict what next value will be
        y_train.append(scaled_data[x, 0])  # 61st value

    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape x train so it works with neural network
    print('1')
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # add one extra dimension - numpy.reshape(a, newshape, order='C')
    print('2')

    # building the model

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # lstm, droupout, lstm, dropout, dense layers

    # ^ experiment with layers and units. the more units and layers the longer you train and risk of overfitting because of too many layers of sophistication
    # lstm is a recurrent cell, so feeds back info not just feed forward like dense layer

    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # prediction of the next closing value

    model.compile(optimizer='adam', loss='mean_squared_error')
    # epochs are meant to be 25
    model.fit(x_train, y_train, epochs=25,
              batch_size=32)  # model will see same data 24 times and model will see 32 units at once

    # see how well this model will perform based on past data that we already have. if i always look back at the last 60 days what are the chances of this model being right

    # TEST THE MODEL ACCURACY ON EXSITING DATA
    # load test data

    test_start = dt.datetime(2021, 1, 1) #start testing earlier in the past so that the user can see how well the model predicts for a longer
    #interval of time
    test_end = dt.datetime.now()  # we have data model has never seen so we can see how well the model performs on data

    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    # create data set that combines training data and test data

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # this is how we load data and have prepared data and now we predict based on the data that we have never seen before to evaluate how accurate the model is

    # MAKE PREDICTIONS ON TEST DATA

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])  # x - prediction days so we dont get negative values

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # tuple of the shape

    predicted_prices = model.predict(x_test)
    # the predicted prices are scaled, so need to reverse scale and inverse transform tem

    predicted_prices = scaler.inverse_transform(predicted_prices)

    # PLOT THE PREDICTIONS TO SEE HOW WELL MODEL PERFORMS
    plt.close()

    fig = Figure(figsize=(5, 4), dpi=100)
    a = fig.add_subplot(111)

    a.plot(actual_prices, color="black", label=f"Actual {company} Price")
    a.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(relx=0.5, rely=0.35, anchor=CENTER)

    clearToolbar()
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
    canvas.draw()
    print("lol")

    # PREDICT THE NEXT DAY

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")
    print("lol")

    var = StringVar()
    label = Message(root, textvariable=var, relief=RAISED)
    label.place(relx=0.5, rely=0.8, anchor=W, width=141, height=40)
    var.set(f"{prediction}")


#####################

menubar = Menu(root)

home = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Main Menu', menu=home)
home.add_separator()
home.add_command(label='Exit', command=root.destroy)

# Adding Stocks menu and commands
stocks = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Stocks', menu=stocks, command=None)
stocks.add_command(label='Google', command=menuOptGoog)

root.config(menu=menubar)

mainloop()
