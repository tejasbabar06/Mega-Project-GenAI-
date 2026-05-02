"""
utils/feature_selection.py
---------------------------
Wraps scikit-learn feature-selection algorithms into a single,
agent-friendly class.

Supported methods
-----------------
  'selectkbest'       – SelectKBest with ANOVA F-score
  'randomforest'      – RandomForestClassifier feature importances
  'lasso'             – Lasso (L1) regularisation via SelectFromModel
  'gradientboosting'  – GradientBoostingClassifier feature importances
  'extratrees'        – ExtraTreesClassifier feature importances
  'mutualinfo'        – Mutual Information scores (model-free)
  'pca'               – PCA dimensionality reduction (returns component labels)

Return format
-------------
Every method now returns a **dict** with keys:
  selected_features  – list of feature name strings
  feature_scores     – dict mapping ALL feature names → importance score
  pca_variance       – list of explained variance ratios (PCA only, else None)
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    SelectFromModel,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


class FeatureSelector:
    """
    Selects the top-K most informative features from a preprocessed dataset.

    Parameters
    ----------
    k      : number of features to select / PCA components (capped at total)
    method : one of the 7 supported method strings (case-insensitive)
    """

    # Map method name → human-readable description (shown in agent logs)
    METHOD_DESCRIPTIONS = {
        "selectkbest":      "SelectKBest (ANOVA F-score)",
        "randomforest":     "RandomForest Importances",
        "lasso":            "Lasso (L1) Regularisation",
        "gradientboosting": "Gradient Boosting Importances",
        "extratrees":       "Extra Trees Importances",
        "mutualinfo":       "Mutual Information Scores",
        "pca":              "PCA Dimensionality Reduction",
    }

    def __init__(self, k: int = 5, method: str = "selectkbest"):
        self.k      = k
        self.method = method.lower().strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_target(y: pd.Series) -> np.ndarray:
        """
        Guarantee the target is a 1-D integer numpy array.
        String / object / category targets are label-encoded automatically.
        """
        if y.dtype == object or str(y.dtype) == "category":
            le = LabelEncoder()
            return le.fit_transform(y.astype(str))
        return y.to_numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Run the chosen feature-selection method and return a result dict.

        Parameters
        ----------
        X : feature matrix (fully numeric after preprocessing)
        y : target series (raw — will be encoded internally if needed)

        Returns
        -------
        dict with keys:
            selected_features  – list of feature names (length ≤ k)
            feature_scores     – dict {col: score} for ALL columns
            pca_variance       – list of explained variance ratios (PCA only)
        """
        # Cap k so we never request more features than exist
        k_actual = min(self.k, X.shape[1])
        y_enc    = self._encode_target(y)

        dispatch = {
            "selectkbest":      self._select_kbest,
            "randomforest":     self._select_rf,
            "lasso":            self._select_lasso,
            "gradientboosting": self._select_gradient_boosting,
            "extratrees":       self._select_extra_trees,
            "mutualinfo":       self._select_mutual_info,
            "pca":              self._select_pca,
        }

        if self.method not in dispatch:
            valid = ", ".join(f"'{m}'" for m in dispatch)
            raise ValueError(
                f"Unknown method '{self.method}'. Choose one of: {valid}."
            )

        return dispatch[self.method](X, y_enc, k_actual)

    # ------------------------------------------------------------------
    # ── Method 1: SelectKBest (ANOVA F-score) ─────────────────────────
    # ------------------------------------------------------------------

    @staticmethod
    def _select_kbest(X: pd.DataFrame, y: np.ndarray, k: int) -> dict:
        """
        SelectKBest with f_classif (ANOVA F-statistic).

        Ranks features by how well their distribution differs across
        classes — good general-purpose univariate filter.
        Best for: classification tasks with numeric features.
        """
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)

        # Build scores dict for ALL features
        scores = dict(zip(X.columns.tolist(), selector.scores_.tolist()))
        selected = list(X.columns[selector.get_support()])

        return {
            "selected_features": selected,
            "feature_scores":    scores,
            "pca_variance":      None,
        }

    # ------------------------------------------------------------------
    # ── Method 2: Random Forest ───────────────────────────────────────
    # ------------------------------------------------------------------

    @staticmethod
    def _select_rf(X: pd.DataFrame, y: np.ndarray, k: int) -> dict:
        """
        RandomForestClassifier Gini impurity importances.

        Fits 100 trees and ranks features by how much they reduce
        impurity on average across all splits.
        Best for: datasets with non-linear relationships.
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        scores = importances.to_dict()
        selected = importances.nlargest(k).index.tolist()

        return {
            "selected_features": selected,
            "feature_scores":    scores,
            "pca_variance":      None,
        }

    # ------------------------------------------------------------------
    # ── Method 3: Lasso (L1 Regularisation) ──────────────────────────
    # ------------------------------------------------------------------

    @staticmethod
    def _select_lasso(X: pd.DataFrame, y: np.ndarray, k: int) -> dict:
        """
        Lasso regression via SelectFromModel.

        Lasso's L1 penalty drives irrelevant feature coefficients exactly
        to zero. LassoCV auto-selects the best regularisation strength α
        via 5-fold cross-validation.

        Because Lasso can eliminate more or fewer than k features,
        we fall back to the top-k by |coefficient| if needed.
        Best for: high-dimensional data, removing redundant features.
        """
        lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
        lasso.fit(X, y)

        coefs = pd.Series(np.abs(lasso.coef_), index=X.columns)
        scores = coefs.to_dict()

        # Primary: features with non-zero coefficients
        sfm      = SelectFromModel(lasso, prefit=True)
        mask     = sfm.get_support()
        selected = list(X.columns[mask])

        # Fallback: if Lasso zeroed everything or too few, use top-k by |coef|
        if len(selected) == 0 or len(selected) < k:
            selected = coefs.nlargest(k).index.tolist()

        # Always return exactly k (or fewer if dataset has fewer cols)
        return {
            "selected_features": selected[:k],
            "feature_scores":    scores,
            "pca_variance":      None,
        }

    # ------------------------------------------------------------------
    # ── Method 4: Gradient Boosting ───────────────────────────────────
    # ------------------------------------------------------------------

    @staticmethod
    def _select_gradient_boosting(X: pd.DataFrame,
                                  y: np.ndarray, k: int) -> dict:
        """
        GradientBoostingClassifier feature importances.

        Builds trees sequentially, each correcting the errors of the last.
        Feature importances reflect how much each feature reduces the loss.
        Best for: structured/tabular data, tends to outperform RandomForest
                  on smaller datasets.
        """
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        gb.fit(X, y)
        importances = pd.Series(gb.feature_importances_, index=X.columns)
        scores = importances.to_dict()
        selected = importances.nlargest(k).index.tolist()

        return {
            "selected_features": selected,
            "feature_scores":    scores,
            "pca_variance":      None,
        }

    # ------------------------------------------------------------------
    # ── Method 5: Extra Trees ─────────────────────────────────────────
    # ------------------------------------------------------------------

    @staticmethod
    def _select_extra_trees(X: pd.DataFrame,
                            y: np.ndarray, k: int) -> dict:
        """
        ExtraTreesClassifier (Extremely Randomised Trees) importances.

        Similar to RandomForest but split thresholds are chosen randomly,
        reducing variance further. Typically faster and can generalise better.
        Best for: large datasets, feature importance with low variance.
        """
        et = ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        et.fit(X, y)
        importances = pd.Series(et.feature_importances_, index=X.columns)
        scores = importances.to_dict()
        selected = importances.nlargest(k).index.tolist()

        return {
            "selected_features": selected,
            "feature_scores":    scores,
            "pca_variance":      None,
        }

    # ------------------------------------------------------------------
    # ── Method 6: Mutual Information ──────────────────────────────────
    # ------------------------------------------------------------------

    @staticmethod
    def _select_mutual_info(X: pd.DataFrame,
                            y: np.ndarray, k: int) -> dict:
        """
        Mutual Information (MI) scores via mutual_info_classif.

        MI measures how much knowing the value of a feature reduces
        uncertainty about the target — captures non-linear dependencies
        that ANOVA F-score misses.
        Best for: datasets with complex, non-linear feature–target relationships.
        """
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns)
        scores = mi_series.to_dict()
        selected = mi_series.nlargest(k).index.tolist()

        return {
            "selected_features": selected,
            "feature_scores":    scores,
            "pca_variance":      None,
        }

    # ------------------------------------------------------------------
    # ── Method 7: PCA ─────────────────────────────────────────────────
    # ------------------------------------------------------------------

    @staticmethod
    def _select_pca(X: pd.DataFrame, y: np.ndarray, k: int) -> dict:
        """
        Principal Component Analysis (PCA) dimensionality reduction.

        PCA projects features into orthogonal directions of maximum variance.
        Unlike other methods here, PCA returns *new* component names
        (PC1, PC2, …) rather than original column names.

        Note: PCA is unsupervised — it ignores the target label.
        Best for: removing multicollinearity, visualisation prep.
        """
        pca = PCA(n_components=k, random_state=42)
        pca.fit(X)

        variance_ratios = pca.explained_variance_ratio_.tolist()

        # Build human-readable component labels with explained variance %
        labels = [
            f"PC{i+1} ({variance_ratios[i]*100:.1f}% var)"
            for i in range(k)
        ]

        # For PCA, feature_scores = loading magnitudes per original feature
        # (sum of absolute loadings across selected components)
        loadings = np.abs(pca.components_).sum(axis=0)
        scores = dict(zip(X.columns.tolist(), loadings.tolist()))

        return {
            "selected_features": labels,
            "feature_scores":    scores,
            "pca_variance":      variance_ratios,
        }
