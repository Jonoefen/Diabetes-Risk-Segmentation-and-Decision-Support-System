import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from flask import send_from_directory
from sklearn.exceptions import NotFittedError


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "Diabetes_and_LifeStyle_Dataset_.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"

ARTIFACT_IMAGES = [
    ("model_comparison.png", "Model comparison (DT / RF / XGBoost)"),
    ("xgboost_confusion_matrix.png", "XGBoost confusion matrix"),
    ("class_imbalance_chart.png", "Class distribution"),
]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _load_required_artifacts():
    xgb_model = joblib.load(ARTIFACTS_DIR / "model_1.pkl")
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    model_features = joblib.load(ARTIFACTS_DIR / "model_features.pkl")

    kmeans_model = joblib.load(ARTIFACTS_DIR / "kmeans.pkl")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    lifestyle_features = joblib.load(ARTIFACTS_DIR / "lifestyle_features.pkl")

    return (
        xgb_model,
        label_encoder,
        model_features,
        kmeans_model,
        scaler,
        lifestyle_features,
    )


def _build_feature_schema():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(
        columns=["diabetes_risk_score", "diagnosed_diabetes", "diabetes_stage"],
        errors="ignore",
    )
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() >= 0.95:
            df[col] = coerced
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    category_options = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in categorical_cols
    }
    numeric_defaults = {}
    for col in numeric_cols:
        median_value = df[col].median(skipna=True)
        numeric_defaults[col] = float(median_value) if pd.notna(median_value) else 0.0
    return categorical_cols, numeric_cols, category_options, numeric_defaults


(
    MODEL,
    LABEL_ENCODER,
    MODEL_FEATURES,
    KMEANS_MODEL,
    SCALER,
    LIFESTYLE_FEATURES,
) = _load_required_artifacts()

(
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    CATEGORY_OPTIONS,
    NUMERIC_DEFAULTS,
) = _build_feature_schema()

FEATURE_IMPORTANCE_DF = pd.read_csv(FEATURE_IMPORTANCE_PATH)

STAGE_RECOMMENDATIONS = {
    "Low Risk": "Maintain current healthy habits, schedule routine yearly screening.",
    "Prediabetes": "Increase activity and improve diet quality; re-test glucose markers in 3-6 months.",
    "Type 1 Diabetes": "Coordinate with clinician for insulin-centered management and close monitoring.",
    "Type 2 Diabetes": "Focus on structured lifestyle plan plus medication adherence with regular HbA1c follow-up.",
    "No Diabetes": "Continue preventive lifestyle habits and routine screening as appropriate.",
}

SEGMENT_RECOMMENDATIONS = {
    0: "Segment 0: Prioritize sustained physical activity and improve sleep consistency.",
    1: "Segment 1: Focus on dietary quality and reducing prolonged screen-time behavior.",
    2: "Segment 2: Emphasize cardiometabolic monitoring and targeted medical follow-up.",
}

_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#f1f5f9", family="Segoe UI, system-ui, sans-serif"),
    margin=dict(l=48, r=24, t=48, b=48),
)


def _shap_bar_figure(top_n=15):
    sub = FEATURE_IMPORTANCE_DF.head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=sub["mean_abs_shap"],
            y=sub["feature"],
            orientation="h",
            marker=dict(color="#14b8a6", line=dict(width=0)),
        )
    )
    fig.update_layout(
        title="Top features — mean |SHAP| (XGBoost)",
        height=max(320, top_n * 28),
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(title="Mean |SHAP|")
    return fig


def _confusion_figure_from_predictions():
    if not PREDICTIONS_PATH.exists() or PREDICTIONS_PATH.stat().st_size < 10:
        return None
    try:
        pred = pd.read_csv(PREDICTIONS_PATH, usecols=["actual_diabetes_stage", "predicted_diabetes_stage"])
    except (ValueError, KeyError):
        return None
    ct = pd.crosstab(pred["actual_diabetes_stage"], pred["predicted_diabetes_stage"])
    fig = go.Figure(
        data=go.Heatmap(
            z=ct.values,
            x=list(ct.columns),
            y=list(ct.index),
            colorscale=[[0, "#0f172a"], [0.5, "#14b8a6"], [1, "#f1f5f9"]],
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title="Hold-out predictions — actual vs predicted (from artifacts)",
        height=420,
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return fig


def _segment_center_figure(segment_id, top_n=10):
    center = KMEANS_MODEL.cluster_centers_[segment_id]
    order = sorted(range(len(LIFESTYLE_FEATURES)), key=lambda i: abs(center[i]), reverse=True)[:top_n]
    labels = [LIFESTYLE_FEATURES[i] for i in order]
    values = [center[i] for i in order]
    colors = ["#14b8a6" if v >= 0 else "#f472b6" for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
        )
    )
    fig.update_layout(
        title=f"Segment {segment_id} — strongest cluster center values (standardized)",
        height=max(300, top_n * 32),
        **_PLOTLY_LAYOUT,
    )
    fig.update_xaxes(title="Cluster center (z)")
    return fig


def _predict_from_inputs(form_values):
    row_df = pd.DataFrame([form_values])

    encoded_row = pd.get_dummies(row_df, drop_first=True)
    encoded_row = encoded_row.reindex(columns=MODEL_FEATURES, fill_value=0)

    class_pred_encoded = MODEL.predict(encoded_row)[0]
    class_pred_label = LABEL_ENCODER.inverse_transform([class_pred_encoded])[0]

    seg_values = row_df[LIFESTYLE_FEATURES]
    seg_scaled = SCALER.transform(seg_values)
    segment_id = int(KMEANS_MODEL.predict(seg_scaled)[0])

    return class_pred_label, segment_id


def _top_classification_drivers(top_n=5):
    top_df = FEATURE_IMPORTANCE_DF.head(top_n)
    return [
        html.Li(f"{row.feature}: {row.mean_abs_shap:.4f}")
        for row in top_df.itertuples(index=False)
    ]


def _top_segment_drivers(segment_id, top_n=5):
    center = KMEANS_MODEL.cluster_centers_[segment_id]
    ranked_idx = sorted(
        range(len(LIFESTYLE_FEATURES)),
        key=lambda idx: abs(center[idx]),
        reverse=True,
    )[:top_n]
    return [
        html.Li(f"{LIFESTYLE_FEATURES[idx]} (z-score center: {center[idx]:.2f})")
        for idx in ranked_idx
    ]


def _numeric_control(column_name):
    return html.Div(
        [
            html.Label(column_name),
            dcc.Input(
                id=f"input-{column_name}",
                type="number",
                value=NUMERIC_DEFAULTS[column_name],
                className="dash-input",
                style={"width": "100%", "padding": "10px 12px", "borderRadius": "8px"},
            ),
        ],
        style={"marginBottom": "4px"},
    )


def _categorical_control(column_name):
    options = [{"label": value, "value": value} for value in CATEGORY_OPTIONS[column_name]]
    default_value = options[0]["value"] if options else None
    return html.Div(
        [
            html.Label(column_name),
            dcc.Dropdown(
                id=f"input-{column_name}",
                options=options,
                value=default_value,
                clearable=False,
                className="dash-dropdown",
            ),
        ],
        style={"marginBottom": "4px"},
    )


def _insight_card_children():
    children = [
        html.Div(
            [
                html.Div(
                    [
                        html.Img(src=f"/artifact-assets/{fname}", alt=caption),
                        html.Div(caption, className="artifact-caption"),
                    ],
                    className="artifact-img",
                )
                for fname, caption in ARTIFACT_IMAGES
            ],
            className="artifact-grid",
        ),
        dcc.Graph(id="shap-graph", figure=_shap_bar_figure(), config={"displayModeBar": True}),
    ]
    confusion_fig = _confusion_figure_from_predictions()
    if confusion_fig is not None:
        children.append(
            dcc.Graph(id="confusion-graph", figure=confusion_fig, config={"displayModeBar": True})
        )
    return children


app = Dash(
    __name__,
    assets_folder=str(_ASSETS_DIR),
    suppress_callback_exceptions=True,
)
app.title = "Diabetes Risk Decision Support"

server = app.server


@server.route("/artifact-assets/<path:filename>")
def serve_artifact_image(filename):
    return send_from_directory(str(ARTIFACTS_DIR), filename)


app.layout = html.Div(
    [
        html.Div(
            [
                html.Div("BC Analytics", className="badge"),
                html.H1("Diabetes Risk Segmentation & Decision Support"),
                html.P(
                    "Explore model insights below, then enter patient features to get a predicted "
                    "diabetes stage, lifestyle segment, and tailored recommendations."
                ),
            ],
            className="app-header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Model insights (from notebooks / artifacts)", className="card-header"),
                        html.Div(_insight_card_children(), className="card-body"),
                    ],
                    className="card",
                ),
                html.Div(
                    [
                        html.Div("Patient input", className="card-header"),
                        html.Div(
                            [
                                html.Div(
                                    [_categorical_control(col) for col in CATEGORICAL_COLUMNS]
                                    + [_numeric_control(col) for col in NUMERIC_COLUMNS],
                                    className="form-grid",
                                ),
                                html.Button(
                                    "Run prediction", id="predict-button", n_clicks=0, className="btn-primary"
                                ),
                                html.Hr(style={"borderColor": "#334155", "margin": "20px 0"}),
                                html.Div(id="prediction-output"),
                            ],
                            className="card-body",
                        ),
                    ],
                    className="card",
                ),
            ],
            className="grid-two",
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [State(f"input-{col}", "value") for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS],
    prevent_initial_call=True,
)
def run_prediction(_n_clicks, *values):
    try:
        all_cols = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
        form_values = dict(zip(all_cols, list(values)))
        if any(v is None for v in form_values.values()):
            return html.Div("Please provide values for all fields.", style={"color": "#f87171"})

        predicted_stage, predicted_segment = _predict_from_inputs(form_values)
        stage_advice = STAGE_RECOMMENDATIONS.get(
            predicted_stage,
            "Follow individualized clinician guidance and lifestyle optimization.",
        )
        segment_advice = SEGMENT_RECOMMENDATIONS.get(
            predicted_segment,
            "Segment recommendation unavailable; review lifestyle metrics manually.",
        )

        seg_fig = _segment_center_figure(predicted_segment)

        return html.Div(
            [
                html.Div(
                    [
                        html.H4(f"Predicted diabetes stage: {predicted_stage}"),
                        html.P(stage_advice),
                    ],
                    className="result-card",
                ),
                html.Div(
                    [
                        html.H4(f"Predicted lifestyle segment: {predicted_segment}"),
                        html.P(segment_advice),
                    ],
                    className="result-card",
                ),
                html.H4(
                    "Key driver analysis",
                    style={"margin": "16px 0 8px", "fontSize": "1.05rem"},
                ),
                html.P(
                    "Top global diabetes-risk drivers (SHAP from classification model):",
                    style={"color": "#94a3b8", "fontSize": "0.9rem"},
                ),
                html.Ul(_top_classification_drivers(top_n=5)),
                html.P(
                    "Top segment characteristics (strongest standardized center values):",
                    style={"color": "#94a3b8", "fontSize": "0.9rem"},
                ),
                html.Ul(_top_segment_drivers(predicted_segment, top_n=5)),
                dcc.Graph(figure=seg_fig, config={"displayModeBar": False}),
            ]
        )
    except (ValueError, KeyError, NotFittedError) as exc:
        return html.Div(f"Prediction error: {exc}", style={"color": "#f87171"})
    except Exception as exc:
        return html.Div(f"Unexpected error: {exc}", style={"color": "#f87171"})


if __name__ == "__main__":
    # Flask's debug reloader exits the parent with code 3; debugpy/VS Code then
    # reports a false failure. Set DASH_USE_RELOADER=1 if you want auto-reload.
    use_reloader = os.environ.get("DASH_USE_RELOADER", "").lower() in ("1", "true", "yes")
    app.run(debug=True, use_reloader=use_reloader)
