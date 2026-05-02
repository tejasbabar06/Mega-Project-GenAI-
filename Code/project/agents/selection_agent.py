"""
Agent 3: FeatureSelectionAgent
--------------------------------
Receives the preprocessing plan, applies it to the raw DataFrame (by
calling the utility modules), and then selects the most important
features using one of 7 available ML-based methods:

  1. selectkbest      – SelectKBest ANOVA F-score (univariate filter)
  2. randomforest     – RandomForest Gini importances
  3. lasso            – Lasso L1 regularisation (SelectFromModel)
  4. gradientboosting – GradientBoosting sequential importances
  5. extratrees       – Extra Trees (extremely randomised) importances
  6. mutualinfo       – Mutual Information (captures non-linear deps)
  7. pca              – PCA dimensionality reduction (returns components)

Returns the processed dataset, selected features, feature importance
scores, and PCA variance ratios (when applicable).
"""

import pandas as pd

from utils.preprocessing import PreprocessingUtils
from utils.feature_selection import FeatureSelector


class FeatureSelectionAgent:
    """
    Simulates an LLM agent that executes preprocessing and selects
    the top-K most informative features.
    """

    def __init__(self, k_features: int = 5, method: str = "selectkbest"):
        """
        Parameters
        ----------
        k_features : number of top features / PCA components (default 5)
        method     : one of 'selectkbest', 'randomforest', 'lasso',
                     'gradientboosting', 'extratrees', 'mutualinfo', 'pca'
        """
        self.name       = "FeatureSelectionAgent"
        self.logs: list[str] = []
        self.k_features = k_features
        self.method     = method

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        self.logs.append(f"[{self.name}] {message}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, df: pd.DataFrame, plan: dict) -> dict:
        """
        Execute the plan and select important features.

        Parameters
        ----------
        df   : the original (raw) DataFrame
        plan : dict produced by FeaturePlanningAgent.plan()

        Returns
        -------
        dict with keys:
            processed_df      – fully preprocessed DataFrame (features only)
            selected_features – list of chosen feature names
            feature_scores    – dict {col: score} for ALL columns
            pca_variance      – list of explained variance ratios (PCA only)
        """
        self.logs.clear()
        self._log("Received preprocessing plan. Starting execution …")

        numeric_cols     = plan["numeric_cols"]
        categorical_cols = plan["categorical_cols"]
        numeric_steps    = plan["numeric_steps"]
        categorical_steps = plan["categorical_steps"]
        target_column    = plan["target_column"]

        # ── Work on a copy so the original is not mutated ─────────────
        processed = df.copy()

        # ── Apply numeric preprocessing ───────────────────────────────
        if numeric_cols:
            if "impute_mean" in numeric_steps:
                self._log("Imputing missing numeric values with column mean …")
                processed = PreprocessingUtils.impute_numeric(processed, numeric_cols)

            if "scale_standard" in numeric_steps:
                self._log("Scaling numeric features with StandardScaler …")
                processed = PreprocessingUtils.scale_numeric(processed, numeric_cols)

        # ── Apply categorical preprocessing ───────────────────────────
        if categorical_cols:
            if "impute_mode" in categorical_steps:
                self._log("Imputing missing categorical values with column mode …")
                processed = PreprocessingUtils.impute_categorical(processed, categorical_cols)

            if "encode_label" in categorical_steps:
                self._log("Encoding categorical features with LabelEncoder …")
                processed = PreprocessingUtils.encode_categorical(processed, categorical_cols)

        # ── Drop the target column from features ──────────────────────
        all_feature_cols = numeric_cols + categorical_cols
        X = processed[all_feature_cols]
        y = df[target_column]           # use raw target (before encoding)

        self._log(f"Feature matrix shape: {X.shape}. "
                  f"Selecting top {self.k_features} features using '{self.method}' …")

        # ── Feature selection ─────────────────────────────────────────
        selector = FeatureSelector(k=self.k_features, method=self.method)
        result   = selector.select(X, y)

        selected_features = result["selected_features"]
        feature_scores    = result["feature_scores"]
        pca_variance      = result["pca_variance"]

        self._log(f"Selected features: {selected_features}")

        # Log top feature scores
        if feature_scores:
            sorted_scores = sorted(feature_scores.items(),
                                   key=lambda x: x[1], reverse=True)
            top_n = min(5, len(sorted_scores))
            self._log(f"Top {top_n} feature scores:")
            for name, score in sorted_scores[:top_n]:
                self._log(f"  → {name}: {score:.4f}")

        if pca_variance:
            total_var = sum(pca_variance) * 100
            self._log(f"PCA total explained variance: {total_var:.1f}%")

        self._log("Feature selection complete.")

        return {
            "processed_df":      X,
            "selected_features": selected_features,
            "feature_scores":    feature_scores,
            "pca_variance":      pca_variance,
        }
