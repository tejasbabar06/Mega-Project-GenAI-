"""
Agent 2: FeaturePlanningAgent
------------------------------
Receives the analysis report from DataAnalysisAgent and decides:
  - Which preprocessing steps to apply to numeric columns
  - Which preprocessing steps to apply to categorical columns
  - The order in which transformations should run

No real LLM is used; decisions are driven by rule-based heuristics that
mirror what a language model would typically recommend.
"""


class FeaturePlanningAgent:
    """
    Simulates an LLM agent that creates a feature-engineering plan.
    """

    def __init__(self):
        self.name = "FeaturePlanningAgent"
        self.logs: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        self.logs.append(f"[{self.name}] {message}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, analysis_report: dict) -> dict:
        """
        Build a preprocessing plan from the analysis report.

        Parameters
        ----------
        analysis_report : dict produced by DataAnalysisAgent.analyse()

        Returns
        -------
        dict with keys:
            numeric_steps    – ordered list of steps for numeric columns
            categorical_steps – ordered list of steps for categorical cols
            numeric_cols     – echoed list of numeric column names
            categorical_cols – echoed list of categorical column names
            target_column    – echoed target column name
        """
        self.logs.clear()
        self._log("Received analysis report. Building preprocessing plan …")

        numeric_cols     = analysis_report["numeric_cols"]
        categorical_cols = analysis_report["categorical_cols"]
        missing_values   = analysis_report["missing_values"]
        target_column    = analysis_report["target_column"]

        # ── Numeric pipeline ──────────────────────────────────────────
        numeric_steps: list[str] = []

        if any(col in missing_values for col in numeric_cols):
            numeric_steps.append("impute_mean")
            self._log("Numeric columns with missing values detected → "
                      "will impute with column mean.")
        else:
            self._log("No missing values in numeric columns → skipping imputation.")

        numeric_steps.append("scale_standard")
        self._log("Will apply StandardScaler to all numeric features.")

        # ── Categorical pipeline ──────────────────────────────────────
        categorical_steps: list[str] = []

        if any(col in missing_values for col in categorical_cols):
            categorical_steps.append("impute_mode")
            self._log("Categorical columns with missing values detected → "
                      "will impute with column mode.")
        else:
            self._log("No missing values in categorical columns → skipping imputation.")

        if categorical_cols:
            categorical_steps.append("encode_label")
            self._log("Will apply LabelEncoder to all categorical features.")
        else:
            self._log("No categorical features present → encoding step skipped.")

        # ── Summary ───────────────────────────────────────────────────
        self._log(f"Numeric pipeline   : {numeric_steps}")
        self._log(f"Categorical pipeline: {categorical_steps}")
        self._log("Plan ready. Passing to FeatureSelectionAgent.")

        plan = {
            "numeric_steps":     numeric_steps,
            "categorical_steps": categorical_steps,
            "numeric_cols":      numeric_cols,
            "categorical_cols":  categorical_cols,
            "target_column":     target_column,
        }
        return plan
