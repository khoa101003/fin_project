import pandas as pd
import numpy as np
import datetime
import yfinance as yf


from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2021-10-29'

df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()

df_raw.to_csv('df_raw.csv')
print(df_raw)

fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list= INDICATORS, use_vix=True, use_turbulence=True, user_defined_feature=False)

processed = fe.preprocess_data(df_raw)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date","tic"]).merge(processed, on=["date","tic"], how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)

def data_split(data, start_date, end_date):
    date_list = data['date'].tolist()
    start_id=0
    end_id=len(data)
    for i in range(len(data)):
        if date_list[i]>start_date:
            start_id=i
            break
    for i in range(i, len(data)):
        if date_list[i]>=end_date:
            while date_list[i]==date_list[i+1]: i+=1
            end_id=i
            break
    
    return data[start_id: end_id+1]


train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)

print(train.close.values)
print(train.date.unique())
train.to_csv('train_data.csv')
trade.to_csv('trade_data.csv')