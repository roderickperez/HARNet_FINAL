import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import date, datetime
import pytz
import numpy as np
utc = pytz.UTC

# def loadStockData(ticker, dateStart, dateEnd):
#     df = yf.download(ticker, start=dateStart, end=dateEnd)
#     df.reset_index(inplace=False)
#     dateStart = df.index[0]
#     dateEnd = df.index[-1]
#     return df, dateStart, dateEnd

def selectDataRanges(df):
    dateStart = df.index[0]
    dateEnd = df.index[-1]
    return dateStart, dateEnd

def loadData(filename, dataSelection, stock, dateStart, dateEnd):

    if dataSelection == 'Yahoo Finance':
        df = yf.download(stock, start=dateStart, end=dateEnd)
        df.index = pd.to_datetime(df.index, infer_datetime_format=True, errors='ignore').date

    elif dataSelection == 'Own Dataset':
        pass
    
    else:
        df = pd.read_csv(filename, sep = ';')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.date
        df.index = df['Date']
        df.drop('Date', axis=1, inplace=True)
            # Remove '.' values for NA
        df = df.replace('.', np.NaN)
        # Delete NA in dataframe
        df = df.dropna()
    
    return df

def convertDatesFormat(dateStart, dateEnd):
    dateStart = pd.to_datetime(dateStart)
    dateEnd = pd.to_datetime(dateEnd)

    my_time = datetime.min.time()

    dateStart = datetime.combine(dateStart, my_time)
    dateEnd = datetime.combine(dateEnd, my_time)

    dateStart = dateStart.replace(tzinfo=utc)
    dateEnd = dateEnd.replace(tzinfo=utc)

    return dateStart, dateEnd

def rangeDataFrame(df, dateStart, dateEnd):
    df = pd.date_range(start=dateStart, end=dateEnd)
    return df
