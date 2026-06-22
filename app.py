"""
app.py — University Admission Trend Analysis Dashboard
Run: streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="University Admission Trend Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"]
FEATURE_LABELS = {
    "GRE_Score": "GRE Score",
    "TOEFL_Score": "TOEFL Score",
    "University_Rating": "University Rating",
    "SOP": "SOP Strength",
    "LOR": "LOR Strength",
    "CGPA": "CGPA",
    "Research": "Research Experience",
}

# ── Load data & models ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("admissions_data.csv")

@st.cache_resource
def load_models():
    lr = joblib.load("logistic_regression.pkl")
    rf = joblib.load("random_forest.pkl")
    linreg = joblib.load("linear_regression.pkl")
    km_bundle = joblib.load("kmeans.pkl")
    return lr, rf, linreg, km_bundle

df = load_data()
lr, rf, linreg, km_bundle = load_models()

X = df[FEATURES]
y_cls = df["Admitted"]
y_reg = df["Chance_of_Admit"]
X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42
)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🎓 Admission Analysis")
st.sidebar.markdown("**Dataset**: Synthetic UCLA-style Graduate Admissions")
st.sidebar.markdown(f"**Records**: {len(df):,}")
st.sidebar.markdown(f"**Admission Rate**: {df['Admitted'].mean():.1%}")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filter")
selected_streams = st.sidebar.multiselect(
    "Stream", options=sorted(df["Stream"].unique()), default=sorted(df["Stream"].unique())
)
year_range = st.sidebar.slider(
    "Year Range", int(df["Year"].min()), int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max()))
)
filtered = df[df["Stream"].isin(selected_streams) & df["Year"].between(*year_range)]

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📈 Trend Analysis", "🤖 ML Models", "🔮 Admission Predictor", "🔵 Clustering", "📂 Batch Upload"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", f"{len(filtered):,}")
    col2.metric("Admission Rate", f"{filtered['Admitted'].mean():.1%}")
    col3.metric("Avg CGPA", f"{filtered['CGPA'].mean():.2f}")
    col4.metric("Avg GRE Score", f"{filtered['GRE_Score'].mean():.0f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(
            filtered, x="CGPA", color="Stream", nbins=30,
            title="CGPA Distribution by Stream",
            labels={"CGPA": "CGPA", "count": "Students"},
            barmode="overlay", opacity=0.7,
        )
        st.plotly_chart(fig, width='stretch')

    with col_b:
        admit_by_stream = (
            filtered.groupby("Stream")["Admitted"].mean().reset_index()
            .rename(columns={"Admitted": "Admission Rate"})
        )
        admit_by_stream["Admission Rate"] = admit_by_stream["Admission Rate"].round(3)
        fig2 = px.bar(
            admit_by_stream, x="Stream", y="Admission Rate",
            title="Admission Rate by Stream",
            color="Stream", text_auto=".1%",
        )
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig2, width='stretch')

    col_c, col_d = st.columns(2)
    with col_c:
        fig3 = px.pie(
            filtered, names="Stream", title="Students by Stream",
            hole=0.4,
        )
        st.plotly_chart(fig3, width='stretch')

    with col_d:
        research_admit = (
            filtered.groupby("Research")["Admitted"].mean().reset_index()
        )
        research_admit["Research"] = research_admit["Research"].map({0: "No Research", 1: "Has Research"})
        fig4 = px.bar(
            research_admit, x="Research", y="Admitted",
            title="Admission Rate: Research vs No Research",
            color="Research", text_auto=".1%",
        )
        fig4.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig4, width='stretch')

    st.subheader("Dataset Sample")
    st.dataframe(filtered.head(20), width='stretch')

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Trend Analysis
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Trend Analysis")

    yearly = (
        filtered.groupby("Year")[["GRE_Score", "TOEFL_Score", "CGPA", "Chance_of_Admit"]]
        .mean()
        .reset_index()
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            yearly, x="Year", y="GRE_Score",
            title="Average GRE Score by Year",
            markers=True, labels={"GRE_Score": "Avg GRE Score"},
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig = px.line(
            yearly, x="Year", y="CGPA",
            title="Average CGPA by Year",
            markers=True, color_discrete_sequence=["#EF553B"],
            labels={"CGPA": "Avg CGPA"},
        )
        st.plotly_chart(fig, width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        fig = px.line(
            yearly, x="Year", y="TOEFL_Score",
            title="Average TOEFL Score by Year",
            markers=True, color_discrete_sequence=["#00CC96"],
            labels={"TOEFL_Score": "Avg TOEFL Score"},
        )
        st.plotly_chart(fig, width='stretch')

    with col4:
        fig = px.line(
            yearly, x="Year", y="Chance_of_Admit",
            title="Average Chance of Admission by Year",
            markers=True, color_discrete_sequence=["#AB63FA"],
            labels={"Chance_of_Admit": "Avg Chance of Admit"},
        )
        st.plotly_chart(fig, width='stretch')

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    num_cols = ["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA",
                "Research", "Chance_of_Admit", "Admitted"]
    corr = filtered[num_cols].corr().round(2)
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig_heat.update_layout(title="Pearson Correlation Matrix", height=500)
    st.plotly_chart(fig_heat, width='stretch')

    # GRE vs CGPA scatter
    st.subheader("GRE vs CGPA — Scatter by Admission")
    fig_sc = px.scatter(
        filtered, x="CGPA", y="GRE_Score",
        color=filtered["Admitted"].map({0: "Not Admitted", 1: "Admitted"}),
        title="GRE Score vs CGPA (coloured by Admission)",
        opacity=0.6,
        color_discrete_map={"Admitted": "#00CC96", "Not Admitted": "#EF553B"},
    )
    st.plotly_chart(fig_sc, width='stretch')

    # University Rating box plots
    st.subheader("Score Distribution by University Rating")
    metric_sel = st.selectbox("Select Metric", ["GRE_Score", "CGPA", "TOEFL_Score", "Chance_of_Admit"])
    fig_box = px.box(
        filtered, x="University_Rating", y=metric_sel,
        color="University_Rating", title=f"{metric_sel} by University Rating",
        labels={"University_Rating": "University Rating"},
    )
    st.plotly_chart(fig_box, width='stretch')

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — ML Models
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("ML Model Performance")

    # Predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    linreg_pred = linreg.predict(X_test)

    lr_acc = accuracy_score(y_cls_test, lr_pred)
    rf_acc = accuracy_score(y_cls_test, rf_pred)
    linreg_r2 = r2_score(y_reg_test, linreg_pred)
    linreg_mae = mean_absolute_error(y_reg_test, linreg_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Logistic Regression Accuracy", f"{lr_acc:.1%}")
    col2.metric("Random Forest Accuracy", f"{rf_acc:.1%}")
    col3.metric("Linear Regression R²", f"{linreg_r2:.4f}")
    col4.metric("Linear Regression MAE", f"{linreg_mae:.4f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Confusion matrices
    def plot_cm(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Not Admitted", "Admitted"],
            y=["Not Admitted", "Admitted"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=False,
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=350,
        )
        return fig

    with col_a:
        st.plotly_chart(plot_cm(y_cls_test, lr_pred, "Logistic Regression — Confusion Matrix"),
                        width='stretch')

    with col_b:
        st.plotly_chart(plot_cm(y_cls_test, rf_pred, "Random Forest — Confusion Matrix"),
                        width='stretch')

    # Feature importance
    st.subheader("Random Forest — Feature Importance")
    importances = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURES],
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        importances, x="Importance", y="Feature",
        orientation="h",
        title="Feature Importance (Random Forest)",
        color="Importance", color_continuous_scale="Viridis",
    )
    fig_fi.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_fi, width='stretch')

    # Linear regression: actual vs predicted
    st.subheader("Linear Regression — Actual vs Predicted (Chance of Admit)")
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(
        x=y_reg_test.values, y=linreg_pred,
        mode="markers", opacity=0.5, name="Predictions",
        marker=dict(color="#636EFA"),
    ))
    lo = min(y_reg_test.min(), linreg_pred.min())
    hi = max(y_reg_test.max(), linreg_pred.max())
    fig_reg.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", name="Perfect Fit",
        line=dict(color="red", dash="dash"),
    ))
    fig_reg.update_layout(
        xaxis_title="Actual Chance of Admit",
        yaxis_title="Predicted Chance of Admit",
        height=400,
    )
    st.plotly_chart(fig_reg, width='stretch')

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Admission Predictor
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🔮 Real-Time Admission Predictor")
    st.markdown("Fill in the student profile below and click **Predict** to get admission probability.")

    with st.form("predictor_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_gre = st.slider("GRE Score", 290, 340, 315)
            p_toefl = st.slider("TOEFL Score", 92, 120, 107)
            p_cgpa = st.slider("CGPA", 6.0, 10.0, 8.5, step=0.1)
        with col2:
            p_uni = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
            p_sop = st.select_slider("SOP Strength", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.5)
            p_lor = st.select_slider("LOR Strength", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.5)
        with col3:
            p_research = st.radio("Research Experience", [0, 1], format_func=lambda x: "Yes" if x else "No", index=1)

        submitted = st.form_submit_button("🔮 Predict Admission", width='stretch')

    if submitted:
        input_data = pd.DataFrame([[p_gre, p_toefl, p_uni, p_sop, p_lor, p_cgpa, p_research]],
                                   columns=FEATURES)

        lr_prob = lr.predict_proba(input_data)[0][1]
        rf_prob = rf.predict_proba(input_data)[0][1]
        linreg_prob_raw = linreg.predict(input_data)[0]
        linreg_prob = float(np.clip(linreg_prob_raw, 0.0, 1.0))
        ensemble_prob = (lr_prob + rf_prob + linreg_prob) / 3

        st.markdown("---")
        st.subheader("Prediction Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Logistic Regression", f"{lr_prob:.1%}")
        c2.metric("Random Forest", f"{rf_prob:.1%}")
        c3.metric("Linear Regression", f"{linreg_prob:.1%}")
        c4.metric("Ensemble (avg)", f"{ensemble_prob:.1%}",
                  delta="Likely Admitted" if ensemble_prob >= 0.65 else "Likely Not Admitted")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ensemble_prob * 100,
            title={"text": "Ensemble Admission Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#636EFA"},
                "steps": [
                    {"range": [0, 40], "color": "#FFCCCC"},
                    {"range": [40, 65], "color": "#FFFACC"},
                    {"range": [65, 100], "color": "#CCFFCC"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 65,
                },
            },
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, width='stretch')

        if ensemble_prob >= 0.65:
            st.success(f"✅ High likelihood of admission ({ensemble_prob:.1%})")
        elif ensemble_prob >= 0.45:
            st.warning(f"⚠️ Borderline case ({ensemble_prob:.1%}) — consider strengthening SOP/LOR or research experience.")
        else:
            st.error(f"❌ Low likelihood of admission ({ensemble_prob:.1%}) — significant improvement needed.")

        # Radar chart of input vs dataset averages
        st.subheader("Your Profile vs Dataset Averages")
        categories = ["GRE (norm)", "TOEFL (norm)", "Uni Rating", "SOP", "LOR", "CGPA (norm)", "Research"]
        student_vals = [
            (p_gre - 290) / 50,
            (p_toefl - 92) / 28,
            (p_uni - 1) / 4,
            (p_sop - 1) / 4,
            (p_lor - 1) / 4,
            (p_cgpa - 6) / 4,
            p_research,
        ]
        avg_vals = [
            (df["GRE_Score"].mean() - 290) / 50,
            (df["TOEFL_Score"].mean() - 92) / 28,
            (df["University_Rating"].mean() - 1) / 4,
            (df["SOP"].mean() - 1) / 4,
            (df["LOR"].mean() - 1) / 4,
            (df["CGPA"].mean() - 6) / 4,
            df["Research"].mean(),
        ]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=student_vals + [student_vals[0]], theta=categories + [categories[0]],
                                             fill="toself", name="Your Profile"))
        fig_radar.add_trace(go.Scatterpolar(r=avg_vals + [avg_vals[0]], theta=categories + [categories[0]],
                                             fill="toself", name="Dataset Average", opacity=0.5))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=400)
        st.plotly_chart(fig_radar, width='stretch')

# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — Clustering
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Student Profile Clustering (KMeans, k=3)")

    if "Cluster" not in df.columns:
        st.warning("Run `python model_trainer.py` first to generate cluster labels.")
    else:
        cluster_order = ["Low Profile", "Mid Profile", "High Profile"]
        color_map = {"Low Profile": "#EF553B", "Mid Profile": "#FFA15A", "High Profile": "#00CC96"}

        col1, col2 = st.columns(2)
        with col1:
            fig_cl = px.scatter(
                df, x="CGPA", y="GRE_Score", color="Cluster",
                title="Clusters: GRE Score vs CGPA",
                opacity=0.7,
                color_discrete_map=color_map,
                category_orders={"Cluster": cluster_order},
            )
            st.plotly_chart(fig_cl, width='stretch')

        with col2:
            fig_cl2 = px.scatter(
                df, x="CGPA", y="Chance_of_Admit", color="Cluster",
                title="Clusters: Chance of Admit vs CGPA",
                opacity=0.7,
                color_discrete_map=color_map,
                category_orders={"Cluster": cluster_order},
            )
            st.plotly_chart(fig_cl2, width='stretch')

        # Cluster stats
        st.subheader("Cluster Statistics")
        cluster_stats = (
            df.groupby("Cluster")[["GRE_Score", "TOEFL_Score", "CGPA", "Chance_of_Admit", "Admitted"]]
            .mean()
            .round(2)
            .reindex(cluster_order)
            .rename(columns={"Admitted": "Admit Rate"})
        )
        st.dataframe(cluster_stats, width='stretch')

        # Cluster distribution bar chart
        cluster_dist = df["Cluster"].value_counts().reset_index()
        cluster_dist.columns = ["Cluster", "Count"]
        fig_dist = px.bar(
            cluster_dist, x="Cluster", y="Count",
            color="Cluster", color_discrete_map=color_map,
            title="Student Count per Cluster",
            text_auto=True,
        )
        st.plotly_chart(fig_dist, width='stretch')

        # 3D scatter
        st.subheader("3D View: GRE × CGPA × Chance of Admit")
        fig_3d = px.scatter_3d(
            df, x="GRE_Score", y="CGPA", z="Chance_of_Admit",
            color="Cluster",
            opacity=0.6,
            title="3D Student Clusters",
            color_discrete_map=color_map,
            category_orders={"Cluster": cluster_order},
        )
        fig_3d.update_layout(height=550)
        st.plotly_chart(fig_3d, width='stretch')

        # Radar for cluster averages
        st.subheader("Cluster Profiles — Radar Chart")
        cats = ["GRE_Score", "TOEFL_Score", "CGPA", "SOP", "LOR"]
        radar_fig = go.Figure()
        for cluster in cluster_order:
            cdf = df[df["Cluster"] == cluster]
            vals = [
                (cdf["GRE_Score"].mean() - 290) / 50,
                (cdf["TOEFL_Score"].mean() - 92) / 28,
                (cdf["CGPA"].mean() - 6) / 4,
                (cdf["SOP"].mean() - 1) / 4,
                (cdf["LOR"].mean() - 1) / 4,
            ]
            labels = ["GRE (norm)", "TOEFL (norm)", "CGPA (norm)", "SOP (norm)", "LOR (norm)"]
            radar_fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=labels + [labels[0]],
                fill="toself", name=cluster,
                line_color=color_map[cluster],
            ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450,
        )
        st.plotly_chart(radar_fig, width='stretch')

# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — Batch Upload
# ══════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("📂 Batch Data Upload & Prediction")
    st.markdown(
        "Upload a CSV of student records to get **batch admission predictions**. "
        "You can then optionally **append** the records to the main dataset."
    )

    # ── Template download ──────────────────────────────────────────────
    template = pd.DataFrame(columns=["GRE_Score", "TOEFL_Score", "University_Rating",
                                      "SOP", "LOR", "CGPA", "Research",
                                      "Chance_of_Admit", "Admitted", "Year", "Stream"])
    example_rows = pd.DataFrame([
        [320, 110, 4, 4.0, 4.0, 8.8, 1, 0.85, 1, 2024, "CS"],
        [300, 98,  2, 2.5, 3.0, 7.2, 0, 0.55, 0, 2023, "EE"],
    ], columns=template.columns)
    template_csv = example_rows.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV Template",
        data=template_csv,
        file_name="batch_template.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── File uploader ──────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        # Validate required feature columns
        missing_cols = [c for c in FEATURES if c not in batch.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        st.success(f"Loaded {len(batch):,} records.")
        st.dataframe(batch.head(10), width='stretch')

        # ── Run predictions ────────────────────────────────────────────
        X_batch = batch[FEATURES]

        batch["LR_Probability"]  = lr.predict_proba(X_batch)[:, 1].round(4)
        batch["RF_Probability"]  = rf.predict_proba(X_batch)[:, 1].round(4)
        linreg_raw               = linreg.predict(X_batch)
        batch["LR_Reg_Score"]    = np.clip(linreg_raw, 0.0, 1.0).round(4)
        batch["Ensemble_Prob"]   = ((batch["LR_Probability"] + batch["RF_Probability"] + batch["LR_Reg_Score"]) / 3).round(4)
        batch["Predicted_Admit"] = (batch["Ensemble_Prob"] >= 0.65).astype(int)

        # Assign KMeans cluster
        km_model  = km_bundle["model"]
        km_scaler = km_bundle["scaler"]
        km_labels = km_bundle["label_map"]
        X_batch_scaled = km_scaler.transform(X_batch)
        batch["Cluster"] = [km_labels[c] for c in km_model.predict(X_batch_scaled)]

        st.markdown("---")
        st.subheader("Prediction Results")

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(batch))
        c2.metric("Predicted Admitted", int(batch["Predicted_Admit"].sum()))
        c3.metric("Predicted Admission Rate", f"{batch['Predicted_Admit'].mean():.1%}")

        st.dataframe(
            batch[["GRE_Score", "TOEFL_Score", "CGPA", "Research",
                   "LR_Probability", "RF_Probability", "Ensemble_Prob",
                   "Predicted_Admit", "Cluster"]],
            width='stretch',
        )

        # Distribution chart
        fig_dist = px.histogram(
            batch, x="Ensemble_Prob", nbins=20,
            color="Cluster",
            title="Ensemble Admission Probability Distribution",
            labels={"Ensemble_Prob": "Ensemble Probability"},
            barmode="overlay", opacity=0.75,
        )
        fig_dist.add_vline(x=0.65, line_dash="dash", line_color="red",
                           annotation_text="Decision Threshold (0.65)")
        st.plotly_chart(fig_dist, width='stretch')

        # Predicted admission by cluster
        fig_cl = px.bar(
            batch.groupby("Cluster")["Predicted_Admit"].mean().reset_index(),
            x="Cluster", y="Predicted_Admit",
            title="Predicted Admission Rate by Cluster",
            color="Cluster", text_auto=".1%",
        )
        fig_cl.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_cl, width='stretch')

        # ── Download results ───────────────────────────────────────────
        results_csv = batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results CSV",
            data=results_csv,
            file_name="batch_predictions.csv",
            mime="text/csv",
        )

        # ── Append to dataset ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("Append to Main Dataset")
        st.markdown(
            "Optionally save these records to `admissions_data.csv`. "
            "If the uploaded file is missing optional columns (Year, Stream, Chance_of_Admit, Admitted) "
            "they will be filled with defaults."
        )

        if st.button("💾 Append Records to admissions_data.csv", type="primary"):
            append_df = batch.copy()

            # Fill optional columns if absent
            if "Year" not in append_df.columns:
                append_df["Year"] = 2024
            if "Stream" not in append_df.columns:
                append_df["Stream"] = "CS"
            if "Chance_of_Admit" not in append_df.columns:
                append_df["Chance_of_Admit"] = append_df["Ensemble_Prob"]
            if "Admitted" not in append_df.columns:
                append_df["Admitted"] = append_df["Predicted_Admit"]

            keep_cols = ["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR",
                         "CGPA", "Research", "Chance_of_Admit", "Admitted", "Year", "Stream", "Cluster"]
            append_df = append_df[[c for c in keep_cols if c in append_df.columns]]

            existing = pd.read_csv("admissions_data.csv")
            combined = pd.concat([existing, append_df], ignore_index=True)
            combined.to_csv("admissions_data.csv", index=False)

            st.success(
                f"✅ Appended {len(append_df):,} records. "
                f"Dataset now has {len(combined):,} rows. "
                "Refresh the page to update all charts."
            )
            st.cache_data.clear()
