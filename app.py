# =================================================================================================
# QTI ENGINEERING WORKBENCH - Definitive Final Edition
#
# AUTHOR: Subject Matter Expert AI
# DATE: 2024-07-23
#
# DESCRIPTION:
# This is the definitive, complete, single-file Streamlit application for a Quality Technical
# Investigation (QTI) Engineer. It incorporates a comprehensive suite of statistical and ML
# methods, alongside enterprise features, using all specified libraries. The data architecture
# uses Streamlit's session state for absolute stability and all known bugs have been corrected.
# =================================================================================================

# --- 1. CORE & UTILITY IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import yaml
import warnings

# --- 2. DATA HANDLING & DATABASE IMPORTS ---
from sqlalchemy import create_engine
import dask.dataframe as dd

# --- 3. VISUALIZATION IMPORTS ---
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from graphviz import Digraph

# --- 4. ANALYTICS & ML IMPORTS ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttost_ind
import shap
from pydantic import BaseModel, Field
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- 5. REPORTING IMPORTS ---
from pptx import Presentation
from pptx.util import Inches

# --- 6. APPLICATION CONFIGURATION ---
st.set_page_config(page_title="QTI Engineering Workbench", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =================================================================================================
# CONFIGURATION & DATA SIMULATION LAYER
# =================================================================================================

class ReportSettings(BaseModel): author: str; company_name: str
class SPCSettings(BaseModel): sigma_level: float = Field(..., ge=2.0, le=4.0)
class AppConfig(BaseModel): report_settings: ReportSettings; spc_settings: SPCSettings

@st.cache_data
def load_config():
    config_string = """
    report_settings: {author: "QTI Engineering Team", company_name: "Diagnostics Inc."}
    spc_settings: {sigma_level: 3.0}
    """
    return AppConfig(**yaml.safe_load(config_string))

@st.cache_data
def generate_process_data(num_records=2000):
    start_time = datetime.now() - timedelta(days=60)
    timestamps = [start_time + timedelta(hours=i) for i in range(num_records)]
    data = []; np.random.seed(42)
    for i, ts in enumerate(timestamps):
        material_lot, operator = f"LOT-{(ts.day % 2) + 1}", f"OP-{(i % 4) + 1}"; is_anomaly = 0
        base_ph, base_vol, base_psi = 7.2, 10.0, 50.0
        if material_lot == 'LOT-2': ph, psi = base_ph + np.random.normal(0.1, 0.1), base_psi + np.random.normal(2.5, 1.0); is_anomaly = 1 if np.random.rand() > 0.85 else 0
        else: ph, psi = base_ph + np.random.normal(0, 0.05), base_psi + np.random.normal(0, 0.5); is_anomaly = 1 if np.random.rand() > 0.98 else 0
        if operator == 'OP-3': vol, psi = base_vol + np.random.normal(0.08, 0.02), psi + 1.5
        elif operator == 'OP-4': vol, psi = base_vol - np.random.normal(0.08, 0.02), psi - 1.5
        else: vol = base_vol + np.random.normal(0, 0.03)
        psi += np.sin(i / 100) * 1.5
        record = {"timestamp": pd.to_datetime(ts), "line_id": f"LINE-{(i % 4) + 1}", "reagent_ph": round(ph, 2), "fill_volume_ml": round(vol, 3), "pressure_psi": round(psi, 1), "operator_id": operator, "material_lot": material_lot, "is_anomaly": is_anomaly}
        data.append(record)
    return pd.DataFrame(data)

@st.cache_data
def generate_capa_data(): return pd.DataFrame([{"id": "CAPA-001", "status": "Closed - Effective", "owner": "Engineer A"}, {"id": "CAPA-002", "status": "Pending VOE", "owner": "Engineer B"}])
@st.cache_data
def generate_method_comparison_data(): np.random.seed(0); true = np.random.uniform(5, 50, 50); return pd.DataFrame({"Old Method": true + np.random.normal(0.5, 1.5, 50), "New Method": true + np.random.normal(0, 1.6, 50)})
@st.cache_data
def generate_lod_data(): conc = np.array([0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]); prob = 1 / (1 + np.exp(-(conc - 1.5) * 1.5)); detected = np.random.binomial(20, prob); return pd.DataFrame({"Concentration": conc, "Detected": detected, "Total": 20})

# =================================================================================================
# MODULE 1: QTI COMMAND CENTER
# =================================================================================================
def show_command_center():
    st.title("🔬 QTI Command Center")
    st.markdown("A real-time overview of the quality system's health, active investigations, and process stability.")
    process_data, capa_data = st.session_state['process_data'], st.session_state['capa_data']
    st.subheader("Key Performance Indicators (KPIs)"); kpi_cols = st.columns(4)
    active_inv = len(capa_data[capa_data['status'] != 'Closed - Effective']); kpi_cols[0].metric("Active Investigations", active_inv)
    kpi_cols[1].metric("New Data Points (24h)", len(process_data[process_data['timestamp'] > datetime.now() - timedelta(days=1)]))
    kpi_cols[2].metric("Avg. Pressure (psi)", f"{process_data['pressure_psi'].mean():.2f}"); kpi_cols[3].metric("Avg. pH", f"{process_data['reagent_ph'].mean():.2f}")
    st.markdown("---"); col1, col2 = st.columns((2, 1.5))
    with col1:
        st.subheader("Process Line Health Status"); line_health = process_data.groupby('line_id')['is_anomaly'].mean().reset_index().rename(columns={'is_anomaly': 'anomaly_rate'})
        fig = px.treemap(line_health, path=['line_id'], values='anomaly_rate', color='anomaly_rate', color_continuous_scale='RdYlGn_r', title="Process Line Anomaly Rate (Lower is Better)")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25)); st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Active CAPA Queue"); st.dataframe(capa_data[capa_data['status'] != 'Closed - Effective'], use_container_width=True, hide_index=True)

# =================================================================================================
# MODULE 2: PROCESS MONITORING
# =================================================================================================
def show_process_monitoring():
    st.title("📈 Process Monitoring & Control")
    st.markdown("Monitor process stability using a suite of univariate and multivariate control charts.")
    process_data = st.session_state['process_data']
    st.sidebar.header("Monitoring Filters"); line_to_monitor = st.sidebar.selectbox("Select Process Line:", sorted(process_data['line_id'].unique()))
    monitor_df = process_data[process_data['line_id'] == line_to_monitor].copy().sort_values('timestamp').reset_index(drop=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Levey-Jennings (I-MR)", "EWMA Chart", "CUSUM Chart", "Multivariate (Hotelling's T²)"])
    with tab1:
        st.subheader("Levey-Jennings (I-MR) Chart"); st.markdown("**Use Case:** Detect large shifts and outliers."); param = st.selectbox("Parameter:", ('reagent_ph', 'fill_volume_ml', 'pressure_psi'), key="imr")
        if len(monitor_df) > 1:
            i_data = monitor_df[param]; i_cl = i_data.mean(); mr = abs(i_data.diff()).dropna(); mr_cl = mr.mean(); sigma = st.session_state['config'].spc_settings.sigma_level
            i_ucl, i_lcl = i_cl + sigma * (mr_cl / 1.128), i_cl - sigma * (mr_cl / 1.128)
            fig = px.line(monitor_df, x='timestamp', y=param, title=f"Individuals Chart for {param}"); fig.add_hline(y=i_cl, line_color='green'); fig.add_hline(y=i_ucl, line_color='red', line_dash='dash'); fig.add_hline(y=i_lcl, line_color='red', line_dash='dash'); st.plotly_chart(fig, use_container_width=True); st.session_state['spc_fig'] = fig
    with tab2:
        st.subheader("EWMA Chart"); st.markdown("**Use Case:** Detect small, sustained process drifts."); param = st.selectbox("Parameter:", ('reagent_ph', 'fill_volume_ml', 'pressure_psi'), key="ewma")
        lambda_val = st.slider("Smoothing Factor (λ)", 0.1, 1.0, 0.2, 0.1);
        if len(monitor_df) > 1:
            ewma = monitor_df[param].ewm(span=(2/lambda_val)-1).mean(); cl, std = monitor_df[param].mean(), monitor_df[param].std(); sigma = st.session_state['config'].spc_settings.sigma_level
            ucl, lcl = cl + sigma * std * np.sqrt(lambda_val / (2 - lambda_val)), cl - sigma * std * np.sqrt(lambda_val / (2 - lambda_val))
            fig = px.line(x=monitor_df['timestamp'], y=ewma, title=f"EWMA Chart (λ={lambda_val})"); fig.add_scatter(x=monitor_df['timestamp'], y=monitor_df[param], mode='lines', name='Raw Data', opacity=0.5)
            fig.add_hline(y=cl, line_color='green'); fig.add_hline(y=ucl, line_color='red', line_dash='dash'); fig.add_hline(y=lcl, line_color='red', line_dash='dash'); st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.subheader("CUSUM Chart"); st.markdown("**Use Case:** Accumulate deviations to detect small, persistent mean shifts."); param = st.selectbox("Parameter:", ('reagent_ph', 'fill_volume_ml', 'pressure_psi'), key="cusum")
        if len(monitor_df) > 1:
            target = monitor_df[param].mean(); cusum = (monitor_df[param] - target).cumsum()
            fig = px.line(x=monitor_df['timestamp'], y=cusum, title=f"CUSUM Chart for {param}", markers=True); fig.add_hline(y=0, line_color='green'); st.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.subheader("Hotelling's T² Chart"); st.markdown("**Use Case:** Monitor correlated variables to detect systemic changes."); features = ['reagent_ph', 'pressure_psi']; X = monitor_df[features]
        if len(X) > len(features):
            X_scaled = StandardScaler().fit_transform(X); inv_cov = np.linalg.inv(np.cov(X_scaled, rowvar=False)); t_sq = [row @ inv_cov @ row.T for row in X_scaled]
            p, n = len(features), len(X); ucl = (p * (n + 1) * (n - 1)) / (n * n - n * p) * stats.f.ppf(0.99, p, n - p)
            fig = px.line(x=monitor_df['timestamp'], y=t_sq, title="Hotelling's T² Chart", markers=True); fig.add_hline(y=ucl, line_color='red', line_dash='dash', name='UCL (99%)'); st.plotly_chart(fig, use_container_width=True)

# =================================================================================================
# MODULE 3: RCA WORKBENCH
# =================================================================================================
def show_rca_workbench():
    st.title("🛠️ Root Cause Analysis (RCA) Workbench")
    process_data = st.session_state['process_data']
    tab1, tab2, tab3, tab4 = st.tabs(["Hypothesis Testing", "Distribution Analysis (KDE)", "Clustering (K-Means)", "AI-Driven Importance"])
    with tab1:
        st.subheader("Statistical Hypothesis Testing"); st.markdown("**Use Case:** Determine if differences between groups are real or random."); param = st.selectbox("Parameter:", ('reagent_ph', 'pressure_psi'), key="ttest")
        g1, g2 = process_data[process_data['material_lot'] == 'LOT-1'][param], process_data[process_data['material_lot'] == 'LOT-2'][param]
        fig = go.Figure(); fig.add_trace(go.Box(y=g1, name='LOT-1')); fig.add_trace(go.Box(y=g2, name='LOT-2')); fig.update_layout(title=f'Comparison of {param} by Lot'); st.plotly_chart(fig, use_container_width=True)
        ttest = stats.ttest_ind(g1, g2, equal_var=False); st.metric("T-test p-value", f"{ttest.pvalue:.4g}");
        if ttest.pvalue < 0.05: st.error("Significant difference detected.")
        else: st.success("No significant difference detected.")
    with tab2:
        st.subheader("Distribution Analysis (KDE)"); st.markdown("**Use Case:** Visualize data distribution to find subtle features like bi-modality."); param = st.selectbox("Parameter:", ('reagent_ph', 'pressure_psi'), key="kde")
        fig = px.violin(process_data, y=param, x='material_lot', color='material_lot', box=True, points="all", title=f"KDE Plot of {param} by Lot"); st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.subheader("Unsupervised Clustering (K-Means)"); st.markdown("**Use Case:** Automatically discover hidden groups in data."); features = ['pressure_psi', 'fill_volume_ml']; X = process_data[features]
        X_scaled = StandardScaler().fit_transform(X); kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled); process_data['cluster'] = kmeans.labels_.astype(str)
        fig = px.scatter(process_data, x=features[0], y=features[1], color='cluster', hover_data=['operator_id', 'material_lot'], title="K-Means Clustering"); st.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.subheader("AI-Driven Root Cause Identification"); st.markdown("**Use Case:** Rank variables by their ability to predict a failure."); model, _ = get_model_and_explainer(process_data); features = ['reagent_ph', 'fill_volume_ml', 'pressure_psi']
        importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        fig = px.bar(importances, x='feature', y='importance', title='Feature Importance for Anomaly Prediction'); st.session_state['rca_importance_fig'] = fig; st.plotly_chart(fig, use_container_width=True)

# =================================================================================================
# MODULE 4: CHANGE VALIDATION & CAPA
# =================================================================================================
def show_change_validation():
    st.title("✅ Change Validation & CAPA Management")
    tab1, tab2, tab3 = st.tabs(["CAPA Log", "Method Agreement (Bland-Altman)", "Equivalence Testing (TOST)"])
    with tab1: st.subheader("CAPA Action Log"); st.dataframe(st.session_state['capa_data'], use_container_width=True, hide_index=True)
    with tab2:
        st.subheader("Method Agreement (Bland-Altman Plot)"); st.markdown("**Use Case:** Visualize bias and limits of agreement between two methods."); df = generate_method_comparison_data(); df['Average'] = df.mean(axis=1); df['Difference'] = df['New Method'] - df['Old Method']
        mean_diff, std_diff = df['Difference'].mean(), df['Difference'].std(); upper_loa, lower_loa = mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff
        fig = px.scatter(df, x='Average', y='Difference', title='Bland-Altman Plot'); fig.add_hline(y=mean_diff, line_color='blue'); fig.add_hline(y=upper_loa, line_color='red', line_dash='dash'); fig.add_hline(y=lower_loa, line_color='red', line_dash='dash'); st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.subheader("Equivalence Testing (TOST)"); st.markdown("**Use Case:** Prove a new process is statistically *equivalent* to an old one."); low, high = st.number_input("Lower Equivalence Bound", -0.5, 0.0, -0.2), st.number_input("Upper Equivalence Bound", 0.0, 0.5, 0.2)
        p_val, _, _ = ttost_ind(np.random.normal(10.0, 1.0, 50), np.random.normal(10.05, 1.0, 50), low=low, upp=high)
        st.metric("TOST p-value", f"{p_val:.4g}");
        if p_val < 0.05: st.success("✅ Processes are statistically equivalent.")
        else: st.error("❌ Equivalence cannot be concluded.")

# =================================================================================================
# MODULE 5: PREDICTIVE & OPTIMIZATION ANALYTICS
# =================================================================================================
@st.cache_resource
def get_model_and_explainer(_df):
    features, target = ['reagent_ph', 'fill_volume_ml', 'pressure_psi'], 'is_anomaly'
    X_train, _, y_train, _ = train_test_split(_df[features], _df[target], test_size=0.3, random_state=42, stratify=_df[target])
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train, y_train)
    return model, shap.TreeExplainer(model)

def show_predictive_and_optimization():
    st.title("🔮 Predictive & Optimization Analytics")
    process_data = st.session_state['process_data']
    tab1, tab2, tab3 = st.tabs(["Time Series Forecasting (SARIMA)", "Process Optimization (Bayesian)", "Real-Time Prediction (XAI)"])
    with tab1:
        st.subheader("Time Series Forecasting (SARIMA)"); st.markdown("**Use Case:** Forecast future process behavior based on historical trends."); ts_data = process_data[['timestamp', 'pressure_psi']].set_index('timestamp').resample('D').mean()
        with st.spinner("Training SARIMA model..."): model = sm.tsa.statespace.SARIMAX(ts_data['pressure_psi'], order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False); forecast = model.get_forecast(steps=30); forecast_df = forecast.summary_frame()
        fig = go.Figure(); fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data['pressure_psi'], name='Historical')); fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], name='Forecast', line=dict(color='orange'))); st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Process Optimization (Bayesian)"); st.markdown("**Use Case:** Intelligently find optimal process settings with minimal experiments.");
        @use_named_args(dimensions=[Real(7.0, 7.4, name='ph'), Real(48.0, 52.0, name='psi')])
        def black_box_function(ph, psi): return -((ph - 7.25)**2 + (psi - 50.5)**2)
        if st.button("Run Bayesian Optimization"):
            with st.spinner("Finding optimal settings..."): res = gp_minimize(black_box_function, dimensions=[Real(7.0, 7.4, name='ph'), Real(48.0, 52.0, name='psi')], n_calls=15, random_state=0)
            st.success(f"Optimization Complete! Optimal settings: pH={res.x[0]:.3f}, Pressure={res.x[1]:.2f} psi.")
    with tab3:
        st.subheader("Real-Time Prediction with XAI"); model, explainer = get_model_and_explainer(process_data); col1, col2 = st.columns([1, 2])
        with col1:
            st.write("Input Parameters:");
            features = ['reagent_ph', 'fill_volume_ml', 'pressure_psi']
            ph, vol, psi = st.slider("pH", 7.0, 7.6, 7.25), st.slider("Volume", 9.8, 10.2, 10.0), st.slider("Pressure", 45.0, 58.0, 51.0);
            input_df = pd.DataFrame([[ph, vol, psi]], columns=features)
            proba = model.predict_proba(input_df)[0][1]; st.metric("Predicted Anomaly Probability", f"{proba:.1%}")
        with col2:
            st.write("XAI Driver Analysis:")
            # DEFINITIVE FIX: Robustly handle SHAP's multi-class output format
            base_value = explainer.expected_value
            shap_values = explainer.shap_values(input_df)
            if isinstance(base_value, list): base_value = base_value[1] # Select value for class 1
            if isinstance(shap_values, list): shap_values = shap_values[1] # Select values for class 1
            
            fig, ax = plt.subplots(figsize=(10, 3)); shap.force_plot(base_value, shap_values[0], input_df.iloc[0], matplotlib=True, show=False); st.pyplot(fig, bbox_inches='tight', dpi=150); plt.close(fig)

# =================================================================================================
# MODULE 6: ADVANCED DEMOS & VALIDATION
# =================================================================================================
def show_advanced_demos():
    st.title("⚙️ Advanced Demos & Validation")
    st.markdown("This module contains self-contained demonstrations of library integrations and specialized validation tasks.")
    tab1, tab2 = st.tabs(["Data Scalability (Dask)", "Assay Validation (LoD)"])
    with tab1:
        st.subheader("Large-Scale Data Processing with Dask"); st.markdown("**Use Case:** Dask enables parallel computation on datasets larger than memory.")
        if st.button("Run Dask Computation"):
            with st.spinner("Processing with Dask..."):
                ddf = dd.from_pandas(st.session_state['process_data'], npartitions=4)
                mean_pressure = ddf.pressure_psi.mean().compute()
            st.success(f"Dask computation complete. Mean pressure across all data: **{mean_pressure:.2f} psi**")
    with tab2:
        st.subheader("Limit of Detection (LoD) by Probit Analysis"); st.markdown("**Use Case:** Determine the lowest concentration an assay can reliably detect."); df = generate_lod_data(); df['Not Detected'] = df['Total'] - df['Detected']
        df['log_conc'] = np.log10(df['Concentration'].replace(0, 0.01));
        glm_binom = sm.GLM(endog=df[['Detected', 'Not Detected']], exog=sm.add_constant(df['log_conc']), family=sm.families.Binomial(link=sm.families.links.probit()))
        res = glm_binom.fit()
        target, params = stats.norm.ppf(0.95), res.params; lod = 10**((target - params['const']) / params['log_conc'])
        st.metric("Calculated Limit of Detection (LoD) at 95%", f"{lod:.3f}");
        x_range = np.linspace(df['log_conc'].min(), df['log_conc'].max(), 200); y_pred = res.predict(sm.add_constant(x_range))
        fig = go.Figure(); fig.add_trace(go.Scatter(x=df['Concentration'], y=df['Detected']/df['Total'], mode='markers', name='Observed')); fig.add_trace(go.Scatter(x=10**x_range, y=y_pred, mode='lines', name='Probit Fit'))
        fig.add_vline(x=lod, line_dash='dash', line_color='red'); fig.add_hline(y=0.95, line_dash='dash', line_color='red')
        fig.update_layout(title='Probit Analysis for Limit of Detection', xaxis_type="log"); st.plotly_chart(fig, use_container_width=True)

# =================================================================================================
# MODULE 7: REPORTING & EXPORT
# =================================================================================================
def show_reporting():
    st.title("📄 Reporting & Export")
    st.markdown("Compile key findings from your investigation into a standardized, downloadable report.")
    if st.button("Generate .pptx Report"):
        with st.spinner("Creating PowerPoint..."):
            config = st.session_state['config']; prs = Presentation()
            # Title Slide
            slide = prs.slides.add_slide(prs.slide_layouts[0]); slide.shapes.title.text = "QTI Investigation Summary"; slide.placeholders[1].text = f"Author: {config.report_settings.author}\nDate: {datetime.now().strftime('%Y%m%d')}"
            # SPC Slide
            if 'spc_fig' in st.session_state:
                slide = prs.slides.add_slide(prs.slide_layouts[5]); slide.shapes.title.text = "Process Monitoring (SPC)"
                img_stream = io.BytesIO(); st.session_state['spc_fig'].write_image(img_stream, format='png'); slide.shapes.add_picture(img_stream, Inches(0.5), Inches(1.5), width=Inches(9))
            # RCA Slide
            if 'rca_importance_fig' in st.session_state:
                slide = prs.slides.add_slide(prs.slide_layouts[5]); slide.shapes.title.text = "AI-Driven Root Cause Analysis"
                img_stream = io.BytesIO(); st.session_state['rca_importance_fig'].write_image(img_stream, format='png'); slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(8))
            # Save and download
            ppt_stream = io.BytesIO(); prs.save(ppt_stream); ppt_stream.seek(0)
            st.download_button("Download Report", ppt_stream, f"QTI_Report_{datetime.now().strftime('%Y%m%d')}.pptx")

# =================================================================================================
# MAIN APPLICATION LOGIC
# =================================================================================================
def main():
    st.sidebar.title("QTI Workbench Navigation")
    st.sidebar.markdown("---")
    # Definitive fix for data persistence: Use st.session_state with simple DataFrames.
    if 'data_loaded' not in st.session_state:
        st.session_state['config'] = load_config()
        st.session_state['process_data'] = generate_process_data()
        st.session_state['capa_data'] = generate_capa_data()
        st.session_state['data_loaded'] = True
    
    page_functions = {
        "QTI Command Center": show_command_center,
        "Process Monitoring": show_process_monitoring,
        "RCA Workbench": show_rca_workbench,
        "Change Validation & CAPA": show_change_validation,
        "Predictive & Optimization": show_predictive_and_optimization,
        "Advanced Demos": show_advanced_demos,
        "Reporting & Export": show_reporting,
    }
    module = st.sidebar.radio("Select a Module:", tuple(page_functions.keys()))
    st.sidebar.markdown("---"); st.sidebar.info("Definitive Edition")
    page_functions[module]()

if __name__ == "__main__":
    main()
