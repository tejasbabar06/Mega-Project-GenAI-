"""
Agent 1: DataAnalysisAgent
--------------------------
Responsible for:
  - Identifying column data types (numeric vs categorical)
  - Detecting missing values per column
  - Producing a structured analysis report used by downstream agents
"""

import pandas as pd


class DataAnalysisAgent:
    """
    Simulates an LLM agent that analyses the uploaded dataset.

    In a real system this reasoning would be driven by a language model.
    Here we use rule-based pandas logic to mimic the same decisions.
    """

    def __init__(self):
        self.name = "DataAnalysisAgent"
        self.logs: list[str] = []          # stores decision logs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Append a timestamped log entry."""
        self.logs.append(f"[{self.name}] {message}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, df: pd.DataFrame, target_column: str) -> dict:
        """
        Analyse *df* and return a structured report.

        Parameters
        ----------
        df            : the raw DataFrame uploaded by the user
        target_column : the column the user wants to predict / use as label

        Returns
        -------
        dict with keys:
            numeric_cols    – list of numeric feature column names
            categorical_cols – list of categorical feature column names
            missing_values  – dict {col: missing_count}
            target_column   – echoed back for downstream agents
            shape           – (rows, cols) of the dataframe
        """
        self.logs.clear()
        self._log(f"Starting analysis on dataset with shape {df.shape}.")

        # ── Step 1: separate features from target ──────────────────────
        feature_cols = [c for c in df.columns if c != target_column]
        self._log(f"Target column identified: '{target_column}'. "
                  f"Remaining feature columns: {len(feature_cols)}.")

        # ── Step 2: classify each feature column ──────────────────────
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []

        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        self._log(f"Numeric features detected   : {numeric_cols}")
        self._log(f"Categorical features detected: {categorical_cols}")

        # ── Step 3: missing-value counts ──────────────────────────────
        missing_values: dict[str, int] = {}
        for col in feature_cols:
            n_missing = int(df[col].isnull().sum())
            if n_missing > 0:
                missing_values[col] = n_missing
                self._log(f"Column '{col}' has {n_missing} missing value(s).")

        if not missing_values:
            self._log("No missing values found in any feature column.")

        # ── Step 4: build report ───────────────────────────────────────
        report = {
            "numeric_cols":     numeric_cols,
            "categorical_cols": categorical_cols,
            "missing_values":   missing_values,
            "target_column":    target_column,
            "shape":            df.shape,
        }

        self._log("Analysis complete. Passing report to FeaturePlanningAgent.")
        return report
