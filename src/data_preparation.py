import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    weekly_sales = df.groupby(['SKU', pd.Grouper(key='DATE', freq='W-MON')])['QUANTITY_SOLD'].sum().reset_index()
    weekly_sales = weekly_sales.sort_values(['SKU', 'DATE'])
    return weekly_sales

def add_time_features(df):
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['week'] = df['DATE'].dt.isocalendar().week
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['DATE'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['week_sin'] = np.sin(2 * np.pi * df['week']/53)
    df['week_cos'] = np.cos(2 * np.pi * df['week']/53)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    return df

def add_lag_features(df, lags=[1, 2, 3, 4, 5, 6, 12, 24, 36, 48, 52]):
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('SKU')['QUANTITY_SOLD'].shift(lag)
    return df

def add_rolling_features(df, windows=[1, 2, 3, 4, 5, 6, 12, 24, 36, 48, 52]):
    df = df.sort_values(['SKU', 'DATE'])
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby('SKU')['QUANTITY_SOLD'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby('SKU')['QUANTITY_SOLD'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    return df

def fill_missing_values(df):
    df = df.sort_values(['SKU', 'DATE'])
    lag_columns = [col for col in df.columns if col.startswith('lag_')]
    rolling_columns = [col for col in df.columns if col.startswith('rolling_')]
    
    for col in lag_columns + rolling_columns:
        df[col] = df.groupby('SKU')[col].ffill()
        df[col] = df[col].fillna(0)
    
    return df

def prepare_features(df):
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = fill_missing_values(df)
    return df

def scale_features(X, scaler_path=None):
    if scaler_path:
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, 'data/scaler.save')
    
    return scaler.transform(X)

def get_feature_columns(df):
    return [col for col in df.columns if col not in ['DATE', 'QUANTITY_SOLD', 'SKU', 'SKU_INDEX', 'CURRENT_LEVEL']]
