import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="Disaster Risk Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _apply_plotly_theme(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    fig.update_layout(
        margin=dict(l=15, r=15, t=50, b=15),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=12))
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig

st.markdown("""
<style>
/* Main Title */
h1 {
    color: var(--primary-color, #ff4b4b) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

/* Tabs */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 24px;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    color: var(--text-color, #555) !important;
    opacity: 0.7;
    font-weight: 600 !important;
    padding-bottom: 8px !important;
    transition: opacity 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary-color, #ff4b4b) !important;
    opacity: 1;
    border-bottom: 2px solid var(--primary-color, #ff4b4b) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    climate = pd.read_csv("climate.csv")
    disaster = pd.read_csv("disaster.csv")
    flood = pd.read_csv("flood.csv")
    return climate, disaster, flood


def _prep_climate(climate: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Country",
        "Year",
        "Avg Temperature (°C)",
        "CO2 Emissions (Tons/Capita)",
        "Sea Level Rise (mm)",
        "Rainfall (mm)",
        "Population",
        "Renewable Energy (%)",
        "Extreme Weather Events",
        "Forest Area (%)",
    ]
    c = climate.copy()
    missing = [k for k in keep if k not in c.columns]
    if missing:
        raise ValueError(f"climate.csv missing columns: {missing}")
    c = c[keep].rename(
        columns={
            "Country": "location",
            "Year": "year",
            "Avg Temperature (°C)": "temperature_c",
            "CO2 Emissions (Tons/Capita)": "co2_tons_per_capita",
            "Sea Level Rise (mm)": "sea_level_rise_mm",
            "Rainfall (mm)": "rainfall_mm",
            "Population": "population",
            "Renewable Energy (%)": "renewable_energy_pct",
            "Extreme Weather Events": "extreme_weather_events",
            "Forest Area (%)": "forest_area_pct",
        }
    )
    c["year"] = pd.to_numeric(c["year"], errors="coerce").astype("Int64")
    c["location"] = c["location"].astype(str).str.strip()
    return c.dropna(subset=["location", "year"])


def _prep_disaster(disaster: pd.DataFrame) -> pd.DataFrame:
    d = disaster.copy()
    keep = ["Country", "Year", "Disaster Type"]
    missing = [k for k in keep if k not in d.columns]
    if missing:
        raise ValueError(f"disaster.csv missing columns: {missing}")
    d = d[keep].rename(columns={"Country": "location", "Year": "year", "Disaster Type": "disaster_type"})
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    d["location"] = d["location"].astype(str).str.strip()
    d["disaster_type"] = d["disaster_type"].astype(str).str.strip()

    # Collapse many events → one label per (location, year)
    agg = (
        d.dropna(subset=["location", "year"])
        .groupby(["location", "year"], as_index=False)
        .agg(
            disaster_occurred=("disaster_type", lambda s: 1),
            disaster_types=("disaster_type", lambda s: ", ".join(sorted(set([x for x in s if x and x != "nan"]))[:8])),
            disaster_events=("disaster_type", "count"),
        )
    )
    return agg


def _merge(climate_p: pd.DataFrame, disaster_p: pd.DataFrame) -> pd.DataFrame:
    merged = climate_p.merge(disaster_p, on=["location", "year"], how="left")
    merged["disaster_occurred"] = merged["disaster_occurred"].fillna(0).astype(int)
    merged["disaster_events"] = merged["disaster_events"].fillna(0).astype(int)
    merged["disaster_types"] = merged["disaster_types"].fillna("None")
    merged = merged.sort_values(["location", "year"]).reset_index(drop=True)
    return merged


climate_raw, disaster_raw, flood_raw = load_data()
climate = _prep_climate(climate_raw)
disaster = _prep_disaster(disaster_raw)
merged = _merge(climate, disaster)


st.title("Disaster Risk Prediction Dashboard")
st.caption("Interactive analysis of climate and disaster patterns.")

with st.sidebar:
    st.subheader("Filters")
    all_countries = sorted(merged["location"].unique().tolist())
    default_countries = all_countries[:10] if len(all_countries) > 10 else all_countries
    countries = st.multiselect("Select Countries", options=all_countries, default=default_countries)

    years = merged["year"].dropna().astype(int)
    yr_min, yr_max = int(years.min()), int(years.max())
    year_range = st.slider("Year Range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))

    st.divider()

    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Choose Algorithm", ["Random Forest", "Logistic Regression", "SVM (RBF)", "KNN", "Decision Tree"]
    )
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

df = merged[merged["location"].isin(countries)].copy() if countries else merged.copy()
df = df[df["year"].between(year_range[0], year_range[1])].copy()


st.subheader("Key Metrics")

total_rows = len(df)
n_countries = df['location'].nunique()
d_years = int(df['disaster_occurred'].sum())
rate = float(df["disaster_occurred"].mean()) if len(df) else 0.0
d_rate = f"{rate*100:.1f}%"

metrics_html = f"""
<div style="display: flex; gap: 1rem; width: 100%; margin-bottom: 2rem;">
    <div style="flex: 1; background: linear-gradient(135deg, #1ea3f6, #0be5f8); border-radius: 8px; padding: 20px; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="margin:0; font-size: 15px; font-weight: 500; color: rgba(255,255,255,0.9);">Total Rows</h4>
        <h2 style="margin:10px 0 0 0; font-size: 34px; font-weight: 600;">{total_rows:,}</h2>
    </div>
    <div style="flex: 1; background: linear-gradient(135deg, #32dca2, #60efb7); border-radius: 8px; padding: 20px; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="margin:0; font-size: 15px; font-weight: 500; color: rgba(255,255,255,0.9);">Countries</h4>
        <h2 style="margin:10px 0 0 0; font-size: 34px; font-weight: 600;">{n_countries:,}</h2>
    </div>
    <div style="flex: 1; background: linear-gradient(135deg, #fa7e7b, #fdac63); border-radius: 8px; padding: 20px; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="margin:0; font-size: 15px; font-weight: 500; color: rgba(255,255,255,0.9);">Disaster Years</h4>
        <h2 style="margin:10px 0 0 0; font-size: 34px; font-weight: 600;">{d_years:,}</h2>
    </div>
    <div style="flex: 1; background: linear-gradient(135deg, #6c5bdc, #8c76e2); border-radius: 8px; padding: 20px; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="margin:0; font-size: 15px; font-weight: 500; color: rgba(255,255,255,0.9);">Disaster Rate</h4>
        <h2 style="margin:10px 0 0 0; font-size: 34px; font-weight: 600;">{d_rate}</h2>
    </div>
</div>
"""
st.markdown(metrics_html, unsafe_allow_html=True)


tab_overview, tab_stats, tab_model, tab_data = st.tabs(["Overview", "Statistics", "Modeling", "Data"])

with tab_overview:
    st.subheader("Overview Analysis")
    st.write("") # small spacing
    
    col_chart1, col_chart2 = st.columns(2)
    
    df_plot = df.copy()
    df_plot["disaster_str"] = df_plot["disaster_occurred"].apply(lambda x: f"{float(x):.1f}")
    
    with col_chart1:
        fig1 = px.histogram(
            df_plot,
            x="temperature_c",
            color="disaster_str",
            title="Temperature Distribution",
            color_discrete_sequence=["#1EA3F6", "#FA7E7B"],
            labels={"temperature_c": "Temperature (°C)", "disaster_str": "Disaster Occurred"},
            hover_data=df_plot.columns
        )
        fig1.update_layout(barmode='stack', legend_title_text="Disaster Occurred", hovermode="x unified")
        fig1 = _apply_plotly_theme(fig1, height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        fig2 = px.scatter(
            df_plot,
            x="rainfall_mm",
            y="temperature_c",
            color="disaster_occurred",
            title="Rainfall vs Temperature",
            color_continuous_scale="Tealgrn",
            labels={"rainfall_mm": "Rainfall (mm)", "temperature_c": "Temperature (°C)", "disaster_occurred": "Likelihood"}
        )
        fig2.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color="rgba(0,0,0,0.2)")))
        fig2.update_layout(hovermode="closest")
        fig2 = _apply_plotly_theme(fig2, height=400)
        st.plotly_chart(fig2, use_container_width=True)

with tab_stats:
    with st.container(border=True):
        st.subheader("Distributions and outliers")

    numeric_cols = [
        "temperature_c",
        "rainfall_mm",
        "co2_tons_per_capita",
        "sea_level_rise_mm",
        "population",
        "renewable_energy_pct",
        "extreme_weather_events",
        "forest_area_pct",
    ]
    available_numeric = [c for c in numeric_cols if c in df.columns]
    feature = st.selectbox("Feature", options=available_numeric, index=0)

    left, right = st.columns([1.25, 1])
    
    df_stats = df.copy()
    df_stats["Disaster Status"] = df_stats["disaster_occurred"].map({0: "No Disaster", 1: "Disaster"})

    with left:
        fig = px.histogram(
            df_stats,
            x=feature,
            color="Disaster Status",
            nbins=40,
            barmode="overlay",
            height=360,
            title=f"Distribution of {feature}",
            color_discrete_sequence=["#1ea3f6", "#fa7e7b"],
        )
        _apply_plotly_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(
            df_stats,
            x="Disaster Status",
            y=feature,
            color="Disaster Status",
            points="outliers",
            height=360,
            title="Outliers by Outcome",
            color_discrete_sequence=["#1ea3f6", "#fa7e7b"],
        )
        _apply_plotly_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Correlation & interpretation")
        corr = df[available_numeric + ["disaster_occurred"]].corr(numeric_only=True)
        figc = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=["#EF4444", "#F3F4F6", "#10B981"],
            zmin=-1,
            zmax=1,
            title="Correlation heatmap (selected data)",
            height=520,
        )
        _apply_plotly_theme(figc, height=520)
        st.plotly_chart(figc, use_container_width=True)

        # Simple interpretation snippet (data-driven)
        target_corr = corr["disaster_occurred"].drop("disaster_occurred").sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        top = target_corr.head(4)
        bullets = "\n".join([f"- **{k}**: correlation with disasters (r={v:.2f})" for k, v in top.items()])
        st.markdown("Most associated variables in this filtered slice (correlation is not causation):\n" + bullets)

with tab_model:
    with st.container(border=True):
        st.subheader("Binary classification: disaster year vs non-disaster year")

        feature_cols = [c for c in available_numeric if c != "population"] + ["population"]
        X = df[feature_cols].copy()
        y = df["disaster_occurred"].astype(int).copy()

        if y.nunique() < 2 or len(df) < 50:
            st.warning("Not enough data (or only one class) in the current filter selection for modeling.")
            st.stop()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
        )

        numeric_features = feature_cols
        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                    numeric_features,
                )
            ]
        )

        if model_name == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=400,
                random_state=int(random_state),
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        elif model_name == "Logistic Regression":
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=int(random_state),
            )
        elif model_name == "SVM (RBF)":
            clf = SVC(
                probability=True,
                class_weight="balanced",
                random_state=int(random_state),
            )
        elif model_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=15)
        else:
            clf = DecisionTreeClassifier(random_state=int(random_state), class_weight="balanced")

        pipe = Pipeline(steps=[("pre", pre), ("model", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) == 2 else np.nan
        ap = average_precision_score(y_test, y_proba) if y_proba is not None else np.nan

        m1, m2, m3 = st.columns(3)
        m1.metric("F1", f"{f1:.3f}")
        m2.metric("ROC-AUC", "—" if np.isnan(roc) else f"{roc:.3f}")
        m3.metric("Avg Precision", "—" if np.isnan(ap) else f"{ap:.3f}")

        st.divider()
        c1, c2 = st.columns([1, 1.15])
        with c1:
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                title="Confusion matrix",
                height=320,
                color_continuous_scale=["#F3F4F6", "#FF4B4B"],
            )
            _apply_plotly_theme(fig_cm, height=320)
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
                fig_roc.update_layout(
                    title="ROC curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=320,
                )
                _apply_plotly_theme(fig_roc, height=320)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("This model doesn't provide probabilities; ROC/PR curves are hidden.")

        st.divider()
        st.subheader("Model interpretation")
        if model_name == "Random Forest":
            rf = pipe.named_steps["model"]
            importances = getattr(rf, "feature_importances_", None)
            if importances is not None:
                imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
                    "importance", ascending=False
                )
                fig_imp = px.bar(
                    imp_df.head(12),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top feature importances",
                    height=420,
                )
                _apply_plotly_theme(fig_imp, height=420)
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.write("For best interpretability, select **Random Forest** to view feature importances.")

        with st.expander("Classification report"):
            st.code(classification_report(y_test, y_pred, digits=3))

with tab_data:
    with st.container(border=True):
        st.subheader("Filtered dataset")
        st.dataframe(df, use_container_width=True, height=420)

        st.download_button(
            "Download filtered data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_disaster_dashboard_data.csv",
            mime="text/csv",
        )