# Stock Analysis and Prediction Tool

This project is a Python-based GUI application for analyzing and predicting Google (GOOG) stock prices. It provides interactive charts and a machine learning-powered prediction tool, all accessible through a user-friendly interface.

## Features

- **Interactive GUI**: Built with Tkinter for easy navigation and use.
- **Stock Data Visualization**: View Google stock data as line or candlestick charts for different time intervals (1 day, 1 week, 1 month, 1 year).
- **Prediction Tool**: Uses an LSTM neural network to predict the next day's closing price based on historical data.
- **User Feedback**: Error messages and progress notifications are provided via pop-up dialogs.

## Technologies Used

- **Python 3**
- **Tkinter**: For the graphical user interface.
- **Matplotlib & mplfinance**: For plotting line and candlestick charts.
- **pandas & pandas_datareader**: For data manipulation and fetching stock data from Yahoo Finance.
- **scikit-learn**: For data preprocessing (MinMaxScaler).
- **TensorFlow/Keras**: For building and training the LSTM neural network.

## How It Works

1. **Select Stock and Chart Type**: Use the menu to choose Google stock and select either a line or candlestick chart.
2. **Choose Time Interval**: Pick a time interval (day, week, month, year) to view the corresponding chart.
3. **Run Prediction Tool**: Click the prediction button to train a model and forecast the next day's closing price. The result is displayed in the GUI.

## Getting Started

1. Install the required Python packages:
   ```bash
   pip install numpy matplotlib pandas pandas_datareader scikit-learn tensorflow mplfinance
   ```
2. Run the application:
   ```bash
   python main.py
   ```

## Notes

- The prediction tool may take some time to run, as it trains a neural network on historical data.
- All data is fetched live from Yahoo Finance using `pandas_datareader`.

## License

This project is for educational purposes.
