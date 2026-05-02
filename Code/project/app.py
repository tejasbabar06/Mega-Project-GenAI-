"""
app.py  –  Main Flask Application
===================================
Entry point for the Multi-Agent Feature Engineering web app.

Routes
------
GET  /                → Home page
GET  /upload          → Upload form
POST /upload          → Handle file upload, show preview + column picker
POST /analyze         → Run DataAnalysisAgent, show analysis page
POST /run-pipeline    → Run the 3-agent pipeline, show results
"""

import os
import json

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)

# ── Agent imports ──────────────────────────────────────────────────────
from agents.data_agent      import DataAnalysisAgent
from agents.planning_agent  import FeaturePlanningAgent
from agents.selection_agent import FeatureSelectionAgent

# ── Visualization imports ──────────────────────────────────────────────
from utils.visualization import VisualizationUtils

# ══════════════════════════════════════════════════════════════════════
# App configuration
# ══════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# Secret key needed for Flask session (store temp data between requests)
app.secret_key = "multi-agent-fe-secret-2024"

# Where uploaded CSVs are saved
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB limit


# ══════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    """Only CSV files are accepted."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "csv"


def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load CSV from *filepath* and return a pandas DataFrame."""
    return pd.read_csv(filepath)


# ══════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════

# ── Home ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the landing page."""
    return render_template("index.html")


# ── Upload (GET) ──────────────────────────────────────────────────────

@app.route("/upload", methods=["GET"])
def upload_form():
    """Show the file-upload form."""
    return render_template("upload.html", columns=None, preview=None)


# ── Upload (POST) ─────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handle CSV upload:
      1. Validate the file.
      2. Save it to the uploads folder.
      3. Read the first 10 rows for a preview.
      4. Pass column names so the user can pick the target.
    """
    # ── Validate ──────────────────────────────────────────────────────
    if "file" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("upload_form"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected. Please choose a CSV file.", "warning")
        return redirect(url_for("upload_form"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Only .csv files are supported.", "danger")
        return redirect(url_for("upload_form"))

    # ── Save ──────────────────────────────────────────────────────────
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "current_dataset.csv")
    file.save(filepath)

    # ── Read and preview ──────────────────────────────────────────────
    df      = load_dataframe(filepath)
    columns = df.columns.tolist()

    # First 10 rows as an HTML table (Bootstrap-styled via template)
    preview_html = df.head(10).to_html(
        classes="table table-sm table-striped",
        index=False,
        border=0,
    )

    # Missing-value summary
    missing_summary = {
        col: int(df[col].isnull().sum())
        for col in columns
        if df[col].isnull().sum() > 0
    }

    # Column type classification
    column_types = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        else:
            column_types[col] = "categorical"

    # Store filename in session so the pipeline route can reload it
    session["dataset_path"] = filepath
    session["columns"]      = columns

    return render_template(
        "upload.html",
        columns=columns,
        column_types=column_types,
        preview=preview_html,
        missing_summary=missing_summary,
        row_count=len(df),
        col_count=len(columns),
    )


# ── Analyze (POST) ────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze_dataset():
    """
    Run DataAnalysisAgent on the uploaded dataset and display
    a detailed analysis page with visualizations.
    """
    target_column = request.form.get("target_column", "").strip()

    filepath = session.get("dataset_path")
    if not filepath or not os.path.exists(filepath):
        flash("Dataset not found. Please upload again.", "danger")
        return redirect(url_for("upload_form"))

    df = load_dataframe(filepath)

    if target_column not in df.columns:
        flash(f"Target column '{target_column}' not found in dataset.", "danger")
        return redirect(url_for("upload_form"))

    # ── Run DataAnalysisAgent ─────────────────────────────────────────
    agent1          = DataAnalysisAgent()
    analysis_report = agent1.analyse(df, target_column)
    logs_agent1     = agent1.logs[:]

    # ── Build column info table ───────────────────────────────────────
    columns = df.columns.tolist()
    column_info = []
    for col in columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        n_missing  = int(df[col].isnull().sum())
        pct_missing = round((n_missing / len(df)) * 100, 1) if len(df) > 0 else 0
        n_unique   = int(df[col].nunique())
        column_info.append({
            "name":        col,
            "type":        "numeric" if is_numeric else "categorical",
            "missing":     n_missing,
            "pct_missing": pct_missing,
            "unique":      n_unique,
            "is_target":   col == target_column,
        })

    # ── Generate missing values chart ─────────────────────────────────
    missing_chart = VisualizationUtils.missing_values_chart(
        analysis_report["missing_values"]
    )

    # ── Dataset summary stats ─────────────────────────────────────────
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    return render_template(
        "analysis.html",
        analysis_report=analysis_report,
        column_info=column_info,
        logs=logs_agent1,
        missing_chart=missing_chart,
        target_column=target_column,
        row_count=len(df),
        col_count=len(columns),
        memory_mb=memory_mb,
        columns=columns,
    )


# ── Run pipeline (POST) ───────────────────────────────────────────────

@app.route("/run-pipeline", methods=["POST"])
def run_pipeline():
    """
    Orchestrate the three-agent pipeline:

        DataAnalysisAgent
              ↓  analysis_report
        FeaturePlanningAgent
              ↓  preprocessing_plan
        FeatureSelectionAgent
              ↓  {processed_df, selected_features, feature_scores, pca_variance}

    All agent logs are collected and displayed on the results page.
    """
    # ── Read form inputs ──────────────────────────────────────────────
    target_column = request.form.get("target_column", "").strip()
    method        = request.form.get("method", "selectkbest")
    k_features    = int(request.form.get("k_features", 5))

    # ── Reload the dataset ────────────────────────────────────────────
    filepath = session.get("dataset_path")
    if not filepath or not os.path.exists(filepath):
        flash("Dataset not found. Please upload again.", "danger")
        return redirect(url_for("upload_form"))

    df = load_dataframe(filepath)

    if target_column not in df.columns:
        flash(f"Target column '{target_column}' not found in dataset.", "danger")
        return redirect(url_for("upload_form"))

    # ══════════════════════════════════════════════════════════════════
    # AGENT 1 – DataAnalysisAgent
    # ══════════════════════════════════════════════════════════════════
    agent1          = DataAnalysisAgent()
    analysis_report = agent1.analyse(df, target_column)
    logs_agent1     = agent1.logs[:]

    # ══════════════════════════════════════════════════════════════════
    # AGENT 2 – FeaturePlanningAgent
    # ══════════════════════════════════════════════════════════════════
    agent2 = FeaturePlanningAgent()
    plan   = agent2.plan(analysis_report)
    logs_agent2 = agent2.logs[:]

    # ══════════════════════════════════════════════════════════════════
    # AGENT 3 – FeatureSelectionAgent
    # ══════════════════════════════════════════════════════════════════
    agent3 = FeatureSelectionAgent(k_features=k_features, method=method)
    result = agent3.select(df, plan)
    logs_agent3 = agent3.logs[:]

    # ── Unpack results ────────────────────────────────────────────────
    processed_df      = result["processed_df"]
    selected_features = result["selected_features"]
    feature_scores    = result["feature_scores"]
    pca_variance      = result["pca_variance"]

    # ── Build HTML tables for the results page ────────────────────────
    processed_preview_html = processed_df.head(10).round(4).to_html(
        classes="table table-sm table-striped table-bordered",
        index=False,
        border=0,
    )

    # ── Generate charts ───────────────────────────────────────────────
    missing_chart = VisualizationUtils.missing_values_chart(
        analysis_report["missing_values"]
    )

    importance_chart = VisualizationUtils.feature_importance_chart(
        feature_scores
    )

    pca_chart = ""
    if pca_variance:
        pca_chart = VisualizationUtils.pca_variance_chart(pca_variance)

    # ── Build feature scores table (sorted descending) ────────────────
    sorted_scores = sorted(feature_scores.items(),
                           key=lambda x: x[1], reverse=True)

    # ── Combine all agent logs ────────────────────────────────────────
    all_logs = logs_agent1 + [""] + logs_agent2 + [""] + logs_agent3

    return render_template(
        "result.html",
        processed_preview=processed_preview_html,
        selected_features=selected_features,
        feature_scores=sorted_scores,
        pca_variance=pca_variance,
        missing_chart=missing_chart,
        importance_chart=importance_chart,
        pca_chart=pca_chart,
        all_logs=all_logs,
        target_column=target_column,
        method=method,
        k_features=k_features,
        numeric_cols=analysis_report["numeric_cols"],
        categorical_cols=analysis_report["categorical_cols"],
        shape=analysis_report["shape"],
    )


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(" Multi-Agent Feature Engineering Framework")
    print(" Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(debug=True)
