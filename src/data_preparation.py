import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Optional


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data by aggregating sales to weekly level.

    Args:
        df (pd.DataFrame): Input dataframe with daily sales data.

    Returns:
        pd.DataFrame: Aggregated weekly sales data.
    """
    # Convert DATE column to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Group by SKU and week, sum the quantities sold
    weekly_sales = df.groupby(['SKU', pd.Grouper(key='DATE', freq='W-MON')])[
        'QUANTITY_SOLD'].sum().reset_index()

    # Sort the resulting dataframe
    weekly_sales = weekly_sales.sort_values(['SKU', 'DATE'])

    return weekly_sales


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with DATE column.

    Returns:
        pd.DataFrame: Dataframe with additional time-based features.
    """
    # Extract various time components
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['week'] = df['DATE'].dt.isocalendar().week
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['DATE'].dt.quarter

    # Create cyclical features for month, week, and day of week
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 53)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 53)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def add_lag_features(df: pd.DataFrame,
                     lags: List[int] = [1, 2, 3, 4, 5, 6, 12, 24, 36, 48,
                                        52]) -> pd.DataFrame:
    """
    Add lagged features to the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        lags (List[int], optional): List of lag values. Defaults to [1, 2, 3, 4, 5, 6, 12, 24, 36, 48, 52].

    Returns:
        pd.DataFrame: Dataframe with additional lag features.
    """
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('SKU')['QUANTITY_SOLD'].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame,
                         windows: List[int] = [1, 2, 3, 4, 5, 6, 12, 24, 36,
                                               48, 52]) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation features to the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        windows (List[int], optional): List of window sizes. Defaults to [1, 2, 3, 4, 5, 6, 12, 24, 36, 48, 52].

    Returns:
        pd.DataFrame: Dataframe with additional rolling features.
    """
    df = df.sort_values(['SKU', 'DATE'])
    for window in windows:
        # Calculate rolling mean
        df[f'rolling_mean_{window}'] = df.groupby('SKU')[
            'QUANTITY_SOLD'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        # Calculate rolling standard deviation
        df[f'rolling_std_{window}'] = df.groupby('SKU')[
            'QUANTITY_SOLD'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in lag and rolling features.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with filled missing values.
    """
    df = df.sort_values(['SKU', 'DATE'])

    # Identify lag and rolling columns
    lag_columns = [col for col in df.columns if col.startswith('lag_')]
    rolling_columns = [col for col in df.columns if col.startswith('rolling_')]

    for col in lag_columns + rolling_columns:
        # Forward fill within each SKU group
        df[col] = df.groupby('SKU')[col].ffill()
        # Fill any remaining NaNs with 0
        df[col] = df[col].fillna(0)

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare all features for the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with all features prepared.
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = fill_missing_values(df)
    return df


def scale_features(X: pd.DataFrame,
                   scaler_path: Optional[str] = None) -> np.ndarray:
    """
    Scale features using StandardScaler.

    Args:
        X (pd.DataFrame): Input features to be scaled.
        scaler_path (Optional[str], optional): Path to load scaler. If None, a new scaler is fitted. Defaults to None.

    Returns:
        np.ndarray: Scaled features.
    """
    if scaler_path:
        # Load existing scaler
        scaler = joblib.load(scaler_path)
    else:
        # Fit new scaler
        scaler = StandardScaler()
        scaler.fit(X)
        # Save the scaler for future use
        joblib.dump(scaler, 'data/scaler.save')

    return scaler.transform(X)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get the list of feature columns, excluding certain columns.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        List[str]: List of feature column names.
    """
    return [col for col in df.columns if
            col not in ['DATE', 'QUANTITY_SOLD', 'SKU', 'SKU_INDEX',
                        'CURRENT_LEVEL']]