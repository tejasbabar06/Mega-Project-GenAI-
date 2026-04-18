"""
utils/preprocessing.py
-----------------------
Stateless utility class with static methods for all preprocessing tasks:
  - Numeric imputation (mean)
  - Categorical imputation (mode)
  - Standard scaling of numeric columns
  - Label encoding of categorical columns

Each method returns a modified copy of the DataFrame so the original
is never mutated.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PreprocessingUtils:
    """
    Collection of static preprocessing helpers.
    Agents call these methods directly; no instantiation needed.
    """

    # ------------------------------------------------------------------
    # Imputation
    # ------------------------------------------------------------------

    @staticmethod
    def impute_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Fill missing numeric values with the column mean.

        Parameters
        ----------
        df      : DataFrame to process
        columns : list of numeric column names to impute

        Returns
        -------
        DataFrame with NaNs in *columns* filled with their respective means.
        """
        df = df.copy()
        for col in columns:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
        return df

    @staticmethod
    def impute_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Fill missing categorical values with the column mode.

        Parameters
        ----------
        df      : DataFrame to process
        columns : list of categorical column names to impute

        Returns
        -------
        DataFrame with NaNs in *columns* filled with their respective modes.
        """
        df = df.copy()
        for col in columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]      # mode() returns a Series
                df[col] = df[col].fillna(mode_val)
        return df

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply sklearn's LabelEncoder to each categorical column.

        Each unique string value is mapped to an integer.
        Unknown values at inference time would need a more robust encoder;
        for this prototype LabelEncoder is sufficient.

        Parameters
        ----------
        df      : DataFrame to process
        columns : list of categorical column names to encode

        Returns
        -------
        DataFrame with string columns replaced by integer codes.
        """
        df = df.copy()
        le = LabelEncoder()
        for col in columns:
            # Convert to string first to handle mixed types gracefully
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    @staticmethod
    def scale_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply StandardScaler (z-score normalisation) to numeric columns.

        After scaling each column has mean ≈ 0 and standard deviation ≈ 1,
        which prevents features with large magnitudes from dominating.

        Parameters
        ----------
        df      : DataFrame to process
        columns : list of numeric column names to scale

        Returns
        -------
        DataFrame with *columns* standardised.
        """
        df = df.copy()
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df
