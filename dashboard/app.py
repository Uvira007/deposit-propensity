"""
Dash app: Data exploration and Model comparison with interactive Plotly visualizations.
Run from project root: python -m dashboard.app   or   python dashboard/app.py
"""
import json
import os
import sys
from pathlib import Path

import dash
from dash import dcc, html, callback, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import numpy as np

RUN_PIPELINE_DISABLED = os.environ.get("RENDER") == "true" or os.environ.get("DISABLE_RUN_PIPELINE", "").lower() in ("1", "true", "yes")
# Project root and config
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, get_paths

# Load config once at startup
CFG = load_config()
PATHS = get_paths(CFG)


def apply_chart_theme(fig, title=None, height=None, **kwargs):
    """Apply a bold, eye-catching theme to Plotly figures."""
    theme = dict(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.98)",
        plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, system-ui, -apple-system, sans-serif", size=13),
        title=dict(font=dict(size=18, color="#0f172a", family="Segoe UI, system-ui, sans-serif")),
        margin=dict(t=64, b=52, l=60, r=44),
        xaxis=dict(showgrid=True, gridcolor="rgba(15,23,42,0.1)", zeroline=False, linecolor="rgba(15,23,42,0.2)", linewidth=1.5),
        yaxis=dict(showgrid=True, gridcolor="rgba(15,23,42,0.1)", zeroline=False, linecolor="rgba(15,23,42,0.2)", linewidth=1.5),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="rgba(15,23,42,0.15)", borderwidth=1.5),
        hoverlabel=dict(bgcolor="#0f172a", font_size=12),
    )
    if title is not None:
        theme["title"] = title if isinstance(title, dict) else dict(text=title, font=dict(size=18, color="#0f172a"))
    if height is not None:
        theme["height"] = height
    fig.update_layout(**{**theme, **kwargs})


DATA_FILENAME = CFG["data"]["filename"]
TARGET_COL = CFG["data"]["target_column"]

# Bank Marketing (UCI) data dictionary: field name, brief description, type, example
DATA_DICTIONARY = [
    {"field": "age", "description": "Client age in years.", "type": "integer", "example": "30"},
    {"field": "job", "description": "Type of job (e.g. admin., blue-collar, management, retired).", "type": "categorical", "example": "management"},
    {"field": "marital", "description": "Marital status (divorced, married, single).", "type": "categorical", "example": "married"},
    {"field": "education", "description": "Education level (e.g. basic.4y, high.school, university.degree).", "type": "categorical", "example": "university.degree"},
    {"field": "default", "description": "Has credit in default?", "type": "binary", "example": "no"},
    {"field": "balance", "description": "Average yearly balance in euros.", "type": "integer", "example": "1500"},
    {"field": "housing", "description": "Has housing loan?", "type": "binary", "example": "yes"},
    {"field": "loan", "description": "Has personal loan?", "type": "binary", "example": "no"},
    {"field": "contact", "description": "Contact communication type (cellular, telephone).", "type": "categorical", "example": "cellular"},
    {"field": "day", "description": "Last contact day of the month (1–31).", "type": "integer", "example": "15"},
    {"field": "month", "description": "Last contact month (e.g. jan, feb, nov).", "type": "categorical", "example": "may"},
    {"field": "duration", "description": "Last contact duration in seconds (often excluded — known only after call).", "type": "integer", "example": "180"},
    {"field": "campaign", "description": "Number of contacts during this campaign for this client.", "type": "integer", "example": "2"},
    {"field": "pdays", "description": "Days since last contact from a previous campaign (-1 if not contacted).", "type": "integer", "example": "-1"},
    {"field": "previous", "description": "Number of contacts before this campaign.", "type": "integer", "example": "0"},
    {"field": "poutcome", "description": "Outcome of previous campaign (failure, nonexistent, success).", "type": "categorical", "example": "unknown"},
    {"field": "y", "description": "Target: has the client subscribed to a term deposit?", "type": "binary", "example": "no"},
]

app = dash.Dash(
    __name__,
    title="Deposit Propensity — Data & Model Insights",
    suppress_callback_exceptions=True,
    external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"],
)
server = app.server


def load_raw_data():
    """Load raw CSV for data exploration. Returns None if missing."""
    csv_path = PATHS["data_dir"] / DATA_FILENAME
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path, sep=";")


def build_numerical_summary(df: pd.DataFrame):
    """Statistical table with key KPIs."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        summary = df.describe(include="all").T.round(2)
    else:
        summary = df.describe(include="all").T.round(2)
    # Add target distribution as KPI (dict so truth check and pie chart work consistently)
    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts()
        target_pct = (vc / len(df) * 100).round(1).to_dict()
    else:
        target_pct = {}
    return summary, target_pct


def build_data_exploration_content():
    """Build Data exploration tab: numerical summary + graphical summary."""
    df = load_raw_data()
    if df is None:
        return html.Div([
            html.P("Raw data not found. Place the CSV in data/raw/ or run the pipeline once.", className="text-muted"),
        ], className="p-4")

    summary, target_pct = build_numerical_summary(df)
    summary_df = summary.reset_index().rename(columns={"index": "feature"})

    # Table with key stats
    table = dash.dash_table.DataTable(
        data=summary_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in summary_df.columns],
        page_size=15,
        style_cell={"textAlign": "left", "padding": "10px"},
        style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": 0}, "backgroundColor": "rgba(52, 152, 219, 0.1)"},
        ],
    )

    # Target distribution — donut with vivid palette
    if target_pct:
        target_fig = px.pie(
            values=list(target_pct.values()),
            names=list(target_pct.keys()),
            title="Target distribution (subscription)",
            color_discrete_sequence=["#06b6d4", "#8b5cf6", "#ec4899", "#f59e0b"],
            hole=0.45,
        )
        target_fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(size=13),
            marker=dict(line=dict(color="#fff", width=2)),
            pull=[0.02] * len(target_pct),
        )
        apply_chart_theme(target_fig, height=360)
    else:
        target_fig = go.Figure().add_annotation(text="No target column", showarrow=False)
        apply_chart_theme(target_fig, height=350)

    # Numeric distributions — bold bars with borders
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        numeric_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique() < 20][:5]
    n_show = min(6, len(numeric_cols))
    # Single palette: all strong, opaque colors that render reliably in subplots (no conditional colors)
    palette = ["#06b6d4", "#6366f1", "#ec4899", "#ea580c", "#059669", "#7c3aed"]
    if n_show > 0:
        fig_hist = make_subplots(rows=2, cols=3, subplot_titles=numeric_cols[:n_show], vertical_spacing=0.12, horizontal_spacing=0.08)
        for i, col in enumerate(numeric_cols[:n_show]):
            r, c = i // 3 + 1, i % 3 + 1
            x_vals = df[col].dropna()
            fig_hist.add_trace(
                go.Histogram(
                    x=x_vals,
                    name=col,
                    marker_color=palette[i % len(palette)],
                    nbinsx=min(50, max(20, int(x_vals.nunique() / 2))),
                ),
                row=r,
                col=c,
            )
        fig_hist.update_layout(bargap=0.15)
        apply_chart_theme(fig_hist, title="Distribution of numeric features", height=420, showlegend=False)
    else:
        fig_hist = go.Figure().add_annotation(text="No numeric columns to plot", showarrow=False)
        apply_chart_theme(fig_hist, height=400)

    # Correlation heatmap — striking diverging palette
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        heatmap = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=[[0, "#6366f1"], [0.5, "#f8fafc"], [1, "#ec4899"]],
            zmin=-1,
            zmax=1,
        )
        heatmap.update_traces(textfont=dict(size=11, color="#0f172a"))
        apply_chart_theme(heatmap, title="Correlation heatmap (numeric features)", height=420)
    else:
        heatmap = go.Figure().add_annotation(text="Need at least 2 numeric columns for correlation", showarrow=False)
        apply_chart_theme(heatmap, height=400)

    return html.Div([
        html.Div([
            html.H4("Numerical summary", className="section-title mb-2"),
            html.P("Key statistics for all features. First row highlighted for quick scan.", className="text-muted small mb-3"),
            html.Div(table, className="chart-card p-3 rounded mb-4"),
        ]),
        html.Div([
            html.H4("Graphical summary", className="section-title mb-2 mt-4"),
            html.P("Target balance and feature distributions.", className="text-muted small mb-3"),
            html.Div([
                dcc.Graph(figure=target_fig, config={"displayModeBar": True, "responsive": True}, className="dashboard-graph"),
            ], className="chart-card p-3 rounded mb-3"),
            html.Div(dcc.Graph(figure=fig_hist, config={"displayModeBar": True, "responsive": True}, className="dashboard-graph"), className="chart-card p-3 rounded mb-3"),
            html.Div([dcc.Graph(figure=heatmap, config={"displayModeBar": True, "responsive": True}, className="dashboard-graph")], className="chart-card p-3 rounded mt-3 mb-4"),
        ]),
    ], className="p-4 tab-content-inner")


def build_data_dictionary_content():
    """Build Data dictionary tab: table of fields with description, type, and example."""
    dict_df = pd.DataFrame(DATA_DICTIONARY)
    table = dash.dash_table.DataTable(
        data=dict_df.to_dict("records"),
        columns=[
            {"name": "Field", "id": "field"},
            {"name": "Description", "id": "description"},
            {"name": "Type", "id": "type"},
            {"name": "Example", "id": "example"},
        ],
        page_size=20,
        style_cell={"textAlign": "left", "padding": "12px"},
        style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"filter_query": f"{{field}} = '{TARGET_COL}'"}, "backgroundColor": "rgba(46, 204, 113, 0.15)", "fontWeight": "500"},
        ],
    )
    return html.Div([
        html.H4("Data dictionary", className="section-title mb-2"),
        html.P("Bank Marketing (UCI) dataset: field names, brief descriptions, types, and example values. Target field is highlighted.", className="text-muted small mb-3"),
        html.Div(table, className="chart-card p-3 rounded mb-4"),
    ], className="p-4 tab-content-inner")


ROC_COLORS = ["#06b6d4", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981"]


def _hex_to_rgba(hex_color, alpha=0.15):
    """Convert #RRGGBB to rgba string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_evaluation_subplots(roc_curves, evaluation_data, selected_models):
    """Build a single figure with 4 subplots: ROC, P-R, Calibration, Cumulative Gains.
    Uses legendgroup so one legend applies to all; generous spacing and legend below.
    """
    models = list(roc_curves.keys()) if selected_models is None else selected_models
    if not models:
        fig = go.Figure()
        fig.add_annotation(text="No model data", showarrow=False)
        apply_chart_theme(fig, height=520)
        return fig

    y_test = np.array(evaluation_data["y_test"])
    n_total = len(y_test)
    n_pos = int(y_test.sum())
    prevalence = y_test.mean()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("ROC Curve", "Precision–Recall Curve", "Calibration Curve", "Cumulative Gains"),
        vertical_spacing=0.22,
        horizontal_spacing=0.14,
    )

    # Baseline traces (one per subplot; common label "Baseline" with single legend entry)
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#94a3b8", width=1.5),
            name="Baseline", legendgroup="Baseline", showlegend=True,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[prevalence, prevalence], mode="lines",
            line=dict(dash="dash", color="#94a3b8", width=1.5),
            name="Baseline", legendgroup="Baseline", showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#94a3b8", width=1.5),
            name="Baseline", legendgroup="Baseline", showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#94a3b8", width=1.5),
            name="Baseline", legendgroup="Baseline", showlegend=False,
        ),
        row=2, col=2,
    )

    for i, name in enumerate(models):
        c = ROC_COLORS[i % len(ROC_COLORS)]
        fill_rgba = _hex_to_rgba(c, 0.14)
        show_leg = True  # show legend only for first occurrence per model (ROC)

        # (1,1) ROC
        data = roc_curves[name]
        fig.add_trace(
            go.Scatter(
                x=data["fpr"], y=data["tpr"], mode="lines", name=name,
                line=dict(color=c, width=2.5), fill="tozeroy", fillcolor=fill_rgba,
                legendgroup=name, showlegend=show_leg,
            ),
            row=1, col=1,
        )
        # (1,2) P-R
        proba = np.array(evaluation_data["models"][name]["y_pred_proba"])
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision, mode="lines",
                name=f"{name} (AP={ap:.3f})",
                line=dict(color=c, width=2), fill="tozerox", fillcolor=fill_rgba,
                legendgroup=name, showlegend=False,
            ),
            row=1, col=2,
        )
        # (2,1) Calibration
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        fig.add_trace(
            go.Scatter(
                x=mean_pred, y=frac_pos, mode="lines+markers", name=name,
                line=dict(color=c, width=2), marker=dict(size=7),
                legendgroup=name, showlegend=False,
            ),
            row=2, col=1,
        )
        # (2,2) Cumulative gains
        order = np.argsort(-proba)
        y_sorted = y_test[order]
        frac_pop = np.arange(1, n_total + 1, dtype=float) / n_total
        frac_captured = np.cumsum(y_sorted) / n_pos if n_pos > 0 else np.zeros(n_total)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([[0], frac_pop]),
                y=np.concatenate([[0], frac_captured]),
                mode="lines", name=name,
                line=dict(color=c, width=2), fill="tozeroy", fillcolor=fill_rgba,
                legendgroup=name, showlegend=False,
            ),
            row=2, col=2,
        )

    # Axis titles
    fig.update_xaxes(title_text="False positive rate", row=1, col=1)
    fig.update_yaxes(title_text="True positive rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_xaxes(title_text="Mean predicted probability", row=2, col=1)
    fig.update_yaxes(title_text="Fraction of positives", row=2, col=1)
    fig.update_xaxes(title_text="Fraction of population", row=2, col=2)
    fig.update_yaxes(title_text="Fraction of positives captured", row=2, col=2)

    fig.update_layout(
        title_text="Model performance metrics",
        height=780,
        margin=dict(t=72, b=140, l=56, r=44),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(15,23,42,0.15)",
            borderwidth=1,
        ),
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.98)",
        plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, system-ui, -apple-system, sans-serif", size=12),
        title=dict(font=dict(size=18, color="#0f172a")),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(15,23,42,0.1)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(15,23,42,0.1)", zeroline=False)
    # Subplot title font
    fig.update_annotations(font=dict(size=14, color="#0f172a"))
    return fig


def build_roc_pr_cal_figures(roc_curves, evaluation_data, selected_models):
    """Build ROC, P-R, calibration, and cumulative gains figures for the given model names. selected_models: list of keys or None for all."""
    models = list(roc_curves.keys()) if selected_models is None else selected_models
    if not models:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    y_test = np.array(evaluation_data["y_test"])
    n_total = len(y_test)
    n_pos = int(y_test.sum())
    prevalence = y_test.mean()

    # ROC
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8", width=1.5), name="Random"))
    for i, name in enumerate(models):
        data = roc_curves[name]
        c = ROC_COLORS[i % len(ROC_COLORS)]
        roc_fig.add_trace(
            go.Scatter(
                x=data["fpr"], y=data["tpr"], mode="lines", name=name,
                line=dict(color=c, width=3), fill="tozeroy",
                fillcolor=f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},0.15)",
            )
        )
    roc_title = "ROC curves — all models" if len(models) > 1 else f"ROC curve — {models[0]}"
    apply_chart_theme(roc_fig, title=roc_title, height=460, xaxis_title="False positive rate", yaxis_title="True positive rate", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # P-R
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines", line=dict(dash="dash", color="#94a3b8", width=1.5), name="No skill"))
    for i, name in enumerate(models):
        proba = np.array(evaluation_data["models"][name]["y_pred_proba"])
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        c = ROC_COLORS[i % len(ROC_COLORS)]
        pr_fig.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} (AP={ap:.3f})",
                line=dict(color=c, width=2.5), fill="tozerox",
                fillcolor=f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},0.12)",
            )
        )
    pr_title = "Precision-Recall curves — all models" if len(models) > 1 else f"Precision-Recall curve — {models[0]}"
    apply_chart_theme(pr_fig, title=pr_title, height=460, xaxis_title="Recall", yaxis_title="Precision", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Calibration
    cal_fig = go.Figure()
    cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8", width=1.5), name="Perfectly calibrated"))
    for i, name in enumerate(models):
        proba = np.array(evaluation_data["models"][name]["y_pred_proba"])
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        c = ROC_COLORS[i % len(ROC_COLORS)]
        cal_fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=name, line=dict(color=c, width=2), marker=dict(size=8)))
    cal_title = "Calibration curve (reliability diagram)" + (f" — {models[0]}" if len(models) == 1 else "")
    apply_chart_theme(cal_fig, title=cal_title, height=420, xaxis_title="Mean predicted probability", yaxis_title="Fraction of positives", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Cumulative gains: fraction of population (x) vs fraction of positives captured (y)
    gains_fig = go.Figure()
    gains_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8", width=1.5), name="Random"))
    for i, name in enumerate(models):
        proba = np.array(evaluation_data["models"][name]["y_pred_proba"])
        order = np.argsort(-proba)
        y_sorted = y_test[order]
        frac_pop = np.arange(1, n_total + 1, dtype=float) / n_total
        frac_captured = np.cumsum(y_sorted) / n_pos if n_pos > 0 else np.zeros(n_total)
        c = ROC_COLORS[i % len(ROC_COLORS)]
        gains_fig.add_trace(
            go.Scatter(
                x=np.concatenate([[0], frac_pop]), y=np.concatenate([[0], frac_captured]),
                mode="lines", name=name,
                line=dict(color=c, width=2.5), fill="tozeroy",
                fillcolor=f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},0.12)",
            )
        )
    gains_title = "Cumulative gains — all models" if len(models) > 1 else f"Cumulative gains — {models[0]}"
    apply_chart_theme(gains_fig, title=gains_title, height=460, xaxis_title="Fraction of population", yaxis_title="Fraction of positives captured", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    return roc_fig, pr_fig, cal_fig, gains_fig


def load_model_artifacts():
    """Load metrics, ROC curves, and evaluation data. Returns (metrics_df, roc_curves, evaluation_data) or (None, None, None)."""
    metrics_path = PATHS["metrics_dir"] / "model_comparison.csv"
    roc_path = PATHS["dashboard_dir"] / "roc_curves.json"
    eval_path = PATHS["dashboard_dir"] / "evaluation_data.json"
    if not metrics_path.exists():
        return None, None, None
    metrics_df = pd.read_csv(metrics_path, index_col=0)
    roc_curves = None
    if roc_path.exists():
        with open(roc_path) as f:
            roc_curves = json.load(f)
    evaluation_data = None
    if eval_path.exists():
        with open(eval_path) as f:
            evaluation_data = json.load(f)
    return metrics_df, roc_curves, evaluation_data


def build_model_comparison_content():
    """Build Model comparison tab: metrics table, ROC curves, confusion matrices, Run pipeline button."""
    metrics_df, roc_curves, evaluation_data = load_model_artifacts()

    if RUN_PIPELINE_DISABLED:
        run_button = html.Div([
            html.Button("Run pipeline (refresh Model comparison)",
                         id="run-pipeline-btn", 
                         n_clicks=0, 
                         className="btn btn-primary btn-lg",
                           disabled=True),
            html.Div("Pipeline execution disabled on this host (request timeout and memory limits on free tier)."
                     "Metrics and curves were generated at deploy time. Re-deploy the app to refresh", 
                     className="mt-2 text-info small",
                     id="run-pipeline-status"),
        ], className="mb-4")
    else:
        run_button = html.Div([
            html.Button("Run pipeline (refresh Model comparison)", id="run-pipeline-btn", n_clicks=0, className="btn btn-primary btn-lg"),
            html.Div(id="run-pipeline-status", className="mt-2 text-muted small"),
        ], className="mb-4")

    if metrics_df is None or roc_curves is None or evaluation_data is None:
        return html.Div([
            html.H4("Model comparison", className="mt-3"),
            html.P("Run the pipeline once to generate metrics and curves. Then click the button below to refresh this tab.", className="text-muted"),
            run_button,
            html.Div(id="model-comparison-body", children=[
                html.P("No model outputs found. Click 'Run pipeline' to train models and populate this tab.", className="text-warning"),
            ]),
        ], className="p-4")

    # Metrics table
    metrics_table = dash.dash_table.DataTable(
        data=metrics_df.reset_index().rename(columns={"index": "model"}).to_dict("records"),
        columns=[{"name": c, "id": c} for c in ["model"] + metrics_df.columns.tolist()],
        style_cell={"textAlign": "left", "padding": "10px"},
        style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
    )

    # Curves: single 2x2 subplot figure (ROC, P-R, Calibration, Cumulative Gains)
    model_names = list(evaluation_data["models"].keys())
    curve_filter_options = [{"label": "All models", "value": "__all__"}] + [{"label": m, "value": m} for m in model_names]
    evaluation_fig = create_evaluation_subplots(roc_curves, evaluation_data, None)

    # Test set class balance (for imbalanced context)
    y_test = np.array(evaluation_data["y_test"])
    n_pos = int(y_test.sum())
    n_neg = len(y_test) - n_pos
    imbalance_ratio = n_neg / n_pos if n_pos > 0 else 0
    class_balance_blurb = html.Div([
        html.H4("Test set class balance", className="section-title mb-2 mt-4"),
        html.P(
            [
                f"Positive (subscription): {n_pos:,} ({100 * n_pos / len(y_test):.1f}%) · ",
                f"Negative: {n_neg:,} ({100 * n_neg / len(y_test):.1f}%) · ",
                html.Span(f"Imbalance ratio (neg/pos): {imbalance_ratio:.1f}", className="text-muted"),
            ],
            className="text-muted small mb-3",
        ),
    ])

    # Confusion matrices
    # Lighter palette: soft lavender to indigo; dark text for readability on all cells
    cmap_cm = [[0, "#eef2ff"], [0.4, "#c7d2fe"], [0.75, "#a5b4fc"], [1, "#818cf8"]]

    def make_confusion_fig(model_name):
        y_pred = np.array(evaluation_data["models"][model_name]["y_pred"])
        cm = confusion_matrix(y_test, y_pred)
        labels = ["No", "Yes"]
        fig = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=cmap_cm,
        )
        fig.update_traces(textfont=dict(size=15, color="#1e293b", family="Segoe UI, sans-serif"))
        apply_chart_theme(fig, title=f"Confusion matrix — {model_name}", height=400, xaxis_title="Predicted", yaxis_title="Actual")
        return fig

    confusion_fig = make_confusion_fig(model_names[0]) if model_names else go.Figure()

    # Best model & SHAP note
    best_name_path = PATHS["models_dir"] / "best_model_name.txt"
    if best_name_path.exists():
        best_name = best_name_path.read_text().strip()
        shap_note = html.Div([
            html.H4("Interpretability (SHAP)", className="mt-4"),
            html.P([f"Best model by ROC-AUC: ", html.Strong(best_name), ". SHAP summary and feature-importance plots are saved in outputs/plots/ (shap_summary_*.png, shap_importance_*.png). Open those files to view."], className="text-muted small"),
        ])
    else:
        shap_note = html.Div()

    body = html.Div([
        html.Div([
            html.H4("Metrics comparison", className="section-title mb-2"),
            html.P("Performance metrics for all tree-based models. Best model is selected by ROC-AUC.", className="text-muted small mb-3"),
            html.Div(metrics_table, className="chart-card p-3 rounded mb-4"),
        ]),
        html.Div([
            html.H4("ROC, Precision–Recall, calibration & cumulative gains", className="section-title mb-2 mt-4"),
            html.P("Filter by model to compare. One legend below applies to all four plots.", className="text-muted small mb-2"),
            html.Div([
                html.Label("Model: ", className="me-2 align-middle"),
                dcc.Dropdown(id="curve-model-dropdown", options=curve_filter_options, value="__all__", clearable=False, className="d-inline-block", style={"minWidth": "180px"}),
            ], className="mb-3"),
            html.Div(
                dcc.Graph(
                    figure=evaluation_fig,
                    config={"displayModeBar": True, "responsive": True},
                    id="evaluation-plots",
                    className="dashboard-graph",
                ),
                className="chart-card p-3 rounded mb-4",
            ),
        ]),
        class_balance_blurb,
        html.Div([
            html.H4("Confusion matrix", className="section-title mb-2 mt-4"),
            html.P("Select a model to view its confusion matrix on the test set.", className="text-muted small mb-3"),
            html.Div([
                dcc.Dropdown(options=model_names, value=model_names[0] if model_names else None, id="confusion-model-dropdown", className="mb-2"),
                dcc.Graph(figure=confusion_fig, config={"displayModeBar": True, "responsive": True}, id="confusion-graph", className="dashboard-graph"),
            ], className="chart-card p-3 rounded mb-4"),
        ]),
        shap_note,
    ], id="model-comparison-body")

    return html.Div([
        html.H4("Model comparison", className="mt-3"),
        run_button,
        body,
    ], className="p-4")


app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks && n_clicks > 0) return true;
        return window.dash_clientside.no_update;
    }
    """,
    Output("pipeline-loading-store", "data"),
    Input("run-pipeline-btn", "n_clicks"),
)

@app.callback(
    Output("evaluation-plots", "figure"),
    Input("curve-model-dropdown", "value"),
)
def update_curves_from_filter(selected):
    """Single filter updates the combined 2x2 evaluation figure."""
    _, roc_curves, evaluation_data = load_model_artifacts()
    if roc_curves is None or evaluation_data is None:
        return go.Figure()
    selected_models = None if selected == "__all__" or not selected else [selected]
    return create_evaluation_subplots(roc_curves, evaluation_data, selected_models)


@app.callback(
    Output("pipeline-loader-overlay", "children"),
    Input("pipeline-loading-store", "data"),
)
def show_pipeline_overlay(loading):
    if loading:
        return PIPELINE_OVERLAY
    return []


@app.callback(
    Output("confusion-graph", "figure"),
    Input("confusion-model-dropdown", "value"),
)
def update_confusion_figure(selected_model):
    _, _, evaluation_data = load_model_artifacts()
    if not evaluation_data or not selected_model:
        return go.Figure()
    y_test = np.array(evaluation_data["y_test"])
    y_pred = np.array(evaluation_data["models"][selected_model]["y_pred"])
    cm = confusion_matrix(y_test, y_pred)
    labels = ["No", "Yes"]
    cmap_cm = [[0, "#eef2ff"], [0.4, "#c7d2fe"], [0.75, "#a5b4fc"], [1, "#818cf8"]]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=cmap_cm,
    )
    fig.update_traces(textfont=dict(size=15, color="#1e293b", family="Segoe UI, sans-serif"))
    apply_chart_theme(fig, title=f"Confusion matrix — {selected_model}", height=400, xaxis_title="Predicted", yaxis_title="Actual")
    return fig


@app.callback(
    Output("run-pipeline-status", "children"),
    Output("model-comparison-body", "children", allow_duplicate=True),
    Output("pipeline-loading-store", "data", allow_duplicate=True),
    Input("run-pipeline-btn", "n_clicks"),
    State("model-comparison-body", "children"),
    prevent_initial_call=True,
)
def run_pipeline_and_refresh(n_clicks, _current_children):
    if n_clicks is None or n_clicks == 0:
        return "", dash.no_update, dash.no_update
    if RUN_PIPELINE_DISABLED:
        return(
            html.Span("Run pipeline is disabled on this host(free tier)",
            className="text-warning"),
            dash.no_update,
            dash.no_update
        )
    from src.run_pipeline import main
    status = html.Span("Running pipeline (this may take 1–2 minutes)...", className="text-info")
    try:
        main()
        status = html.Span("Pipeline finished. Model comparison tab updated below.", className="text-success")
    except Exception as e:
        status = html.Span(f"Pipeline failed: {e}", className="text-danger")
    # Rebuild model comparison body from fresh artifacts
    metrics_df, roc_curves, evaluation_data = load_model_artifacts()
    if metrics_df is None or roc_curves is None or evaluation_data is None:
        new_body = html.P("No model outputs found after run. Check pipeline logs.", className="text-warning")
        return status, new_body, False
    # Rebuild metrics table and ROC and confusion block
    metrics_table = dash.dash_table.DataTable(
        data=metrics_df.reset_index().rename(columns={"index": "model"}).to_dict("records"),
        columns=[{"name": c, "id": c} for c in ["model"] + metrics_df.columns.tolist()],
        style_cell={"textAlign": "left", "padding": "10px"},
        style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
    )
    evaluation_fig = create_evaluation_subplots(roc_curves, evaluation_data, None)
    curve_filter_options = [{"label": "All models", "value": "__all__"}] + [{"label": m, "value": m} for m in evaluation_data["models"]]
    y_test = np.array(evaluation_data["y_test"])
    n_pos, n_neg = int(y_test.sum()), len(y_test) - int(y_test.sum())
    imbalance_ratio = n_neg / n_pos if n_pos > 0 else 0
    class_balance_blurb = html.Div([html.H4("Test set class balance", className="section-title mb-2 mt-4"), html.P([f"Positive (subscription): {n_pos:,} ({100 * n_pos / len(y_test):.1f}%) · ", f"Negative: {n_neg:,} ({100 * n_neg / len(y_test):.1f}%) · ", html.Span(f"Imbalance ratio (neg/pos): {imbalance_ratio:.1f}", className="text-muted")], className="text-muted small mb-3")])
    model_names = list(evaluation_data["models"].keys())
    sel = model_names[0] if model_names else None
    y_pred = np.array(evaluation_data["models"][sel]["y_pred"]) if sel else np.array([])
    if len(y_pred):
        cm = confusion_matrix(y_test, y_pred)
        cmap_cm = [[0, "#eef2ff"], [0.4, "#c7d2fe"], [0.75, "#a5b4fc"], [1, "#818cf8"]]
        cf_fig = px.imshow(cm, x=["No", "Yes"], y=["No", "Yes"], text_auto=True, aspect="auto", color_continuous_scale=cmap_cm)
        cf_fig.update_traces(textfont=dict(size=15, color="#1e293b", family="Segoe UI, sans-serif"))
        apply_chart_theme(cf_fig, title=f"Confusion matrix — {sel}", height=400, xaxis_title="Predicted", yaxis_title="Actual")
    else:
        cf_fig = go.Figure()
    best_name_path = PATHS["models_dir"] / "best_model_name.txt"
    if best_name_path.exists():
        best_name = best_name_path.read_text().strip()
        shap_note = html.Div([html.H4("Interpretability (SHAP)", className="mt-4"), html.P([f"Best model by ROC-AUC: ", html.Strong(best_name), ". SHAP plots in outputs/plots/."], className="text-muted small")])
    else:
        shap_note = html.Div()
    new_body = html.Div([
        html.Div([html.H4("Metrics comparison", className="section-title mb-2"), html.P("Performance metrics for all tree-based models.", className="text-muted small mb-3"), html.Div(metrics_table, className="chart-card p-3 rounded mb-4")]),
        html.Div([
            html.H4("ROC, Precision-Recall, calibration & cumulative gains", className="section-title mb-2 mt-4"),
            html.P("Filter by model to compare. One legend below applies to all four plots.", className="text-muted small mb-2"),
            html.Div([html.Label("Model: ", className="me-2 align-middle"), dcc.Dropdown(id="curve-model-dropdown", options=curve_filter_options, value="__all__", clearable=False, className="d-inline-block", style={"minWidth": "180px"})], className="mb-3"),
            html.Div(dcc.Graph(figure=evaluation_fig, config={"displayModeBar": True, "responsive": True}, id="evaluation-plots", className="dashboard-graph"), className="chart-card p-3 rounded mb-4"),
        ]),
        class_balance_blurb,
        html.Div([html.H4("Confusion matrix", className="section-title mb-2 mt-4"), html.P("Select a model below to view its confusion matrix.", className="text-muted small mb-3"), html.Div([dcc.Dropdown(options=model_names, value=sel, id="confusion-model-dropdown", className="mb-2"), dcc.Graph(figure=cf_fig, config={"displayModeBar": True, "responsive": True}, id="confusion-graph", className="dashboard-graph")], className="chart-card p-3 rounded mb-4")]),
        shap_note,
    ])
    return status, new_body, False


# Custom pipeline overlay: sits on top of page so background stays visible (dimmed/blurred)
PIPELINE_OVERLAY = html.Div(
    [
        html.Div(
            [
                html.Div(className="pipeline-overlay-spinner"),
                html.P("Running pipeline…", className="pipeline-overlay-msg mt-3 mb-1 fw-semibold"),
                html.P("This may take 1–2 minutes.", className="pipeline-overlay-sub small text-muted"),
            ],
            className="pipeline-overlay-content",
        ),
    ],
    className="pipeline-overlay",
)

# Layout
app.layout = html.Div([
    dcc.Store(id="pipeline-loading-store", data=False),
    html.Div([
        html.H2("Deposit Propensity — Data & Model Insights", className="text-white mb-0"),
        html.P("Explore the Bank Marketing dataset and compare tree-based models.", className="text-white-50 mb-0"),
    ], className="bg-dark p-4 rounded-top"),
    dcc.Tabs(id="tabs", value="dictionary", className="dashboard-tabs mt-3", children=[
        dcc.Tab(label="Data dictionary", value="dictionary", children=[
            html.Div(id="data-dictionary-content"),
        ]),
        dcc.Tab(label="Data exploration", value="data", children=[
            html.Div(id="data-exploration-content"),
        ]),
        dcc.Tab(label="Model comparison", value="models", children=[
            html.Div(id="model-comparison-content"),
        ]),
    ]),
    html.Div(id="pipeline-loader-overlay"),
], className="container-fluid px-4 pb-4")


@app.callback(
    Output("data-exploration-content", "children"),
    Input("tabs", "value"),
)
def serve_data_tab(tab):
    if tab != "data":
        return dash.no_update
    return build_data_exploration_content()


@app.callback(
    Output("data-dictionary-content", "children"),
    Input("tabs", "value"),
)
def serve_dictionary_tab(tab):
    if tab != "dictionary":
        return dash.no_update
    return build_data_dictionary_content()


@app.callback(
    Output("model-comparison-content", "children"),
    Input("tabs", "value"),
)
def serve_models_tab(tab):
    if tab != "models":
        return dash.no_update
    return build_model_comparison_content()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
