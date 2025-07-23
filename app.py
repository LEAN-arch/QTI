# app.py - QTI Engineering Workbench (Corrected Version)

# =================================================================================================
# CHUNK 1: FOUNDATION - IMPORTS, APP STRUCTURE, AND DATA SIMULATION
# =================================================================================================

# --- 1. Core Library Imports ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt # FIX: Import matplotlib directly for use with pyspc

# --- 2. Advanced Analytics & ML Library Imports ---
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline # FIX: Import Pipeline
from pydantic import BaseModel, Field
import scipy.stats as stats
import shap
import pyspc
from graphviz import Digraph

# --- 3. Reporting & Configuration Library Imports ---
from pptx import Presentation
from pptx.util import Inches

# --- 4. Page Configuration (Do this once at the top) ---
st.set_page_config(
    page_title="QTI Engineering Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 5. Data Simulation & Models (Simulating a real-world environment) ---
# Pydantic models for data validation and structure
class ProcessData(BaseModel):
    timestamp: datetime
    process_id: str
    line_id: str
    reagent_ph: float = Field(..., ge=6.8, le=7.8)
    fill_volume_ml: float = Field(..., ge=9.8, le=10.2)
    pressure_psi: float = Field(..., ge=45.0, le=55.0)
    operator_id: str
    batch_id: str
    material_lot: str
    is_anomaly: int = Field(..., ge=0, le=1)

@st.cache_data
def generate_process_data(num_records=2000):
    """Generates realistic, multivariate process data for monitoring and analysis."""
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=i*15) for i in range(num_records)]
    
    data = []
    # Create a systemic issue for a specific material lot
    for ts in timestamps:
        material_lot = f"LOT-{(ts.day % 2) + 1}"
        is_anomaly = 0
        
        base_ph = 7.2
        base_vol = 10.0
        base_psi = 50.0
        
        # Introduce a correlation: LOT-2 has higher pressure and pH
        if material_lot == 'LOT-2':
            ph = base_ph + np.random.normal(0.1, 0.1) # Shifted pH
            psi = base_psi + np.random.normal(2.5, 1.0) # Shifted Pressure
            if np.random.rand() > 0.85: # Higher chance of anomaly for this lot
                is_anomaly = 1
        else: # LOT-1 is stable
            ph = base_ph + np.random.normal(0, 0.05)
            psi = base_psi + np.random.normal(0, 0.5)
            if np.random.rand() > 0.98:
                is_anomaly = 1
        
        vol = base_vol + np.random.normal(0, 0.03)

        record = {
            "timestamp": ts,
            "process_id": f"PROC-{(ts.day % 3) + 1}",
            "line_id": f"LINE-{(ts.hour % 4) + 1}",
            "reagent_ph": round(ph, 2),
            "fill_volume_ml": round(vol, 3),
            "pressure_psi": round(psi, 1),
            "operator_id": f"OP-{(ts.day % 5) + 1}",
            "batch_id": f"B-{ts.strftime('%Y%m%d')}-{((ts.hour // 8) % 3) + 1}",
            "material_lot": material_lot,
            "is_anomaly": is_anomaly
        }
        data.append(record)
    
    return pd.DataFrame(data)

@st.cache_data
def generate_complaint_data():
    """Generates mock customer complaint data for NLP analysis."""
    complaints = [
        {"id": "C-001", "text": "The reagent in lot LOT-2 seems to be giving false positives. We had to rerun the entire batch.", "category": "Reagent"},
        {"id": "C-002", "text": "Received an instrument with a leaking fill tube. The pressure seems unstable.", "category": "Hardware"},
        {"id": "C-003", "text": "Software crashed mid-run, showing error code 502. Lost all data for the current plate.", "category": "Software"},
        {"id": "C-004", "text": "The pH level for several kits from LOT-2 were out of spec upon arrival.", "category": "Reagent"},
        {"id": "C-005", "text": "Machine #3 is consistently showing higher pressure readings than the others, causing seal failures.", "category": "Hardware"}
    ]
    return pd.DataFrame(complaints)

@st.cache_data
def generate_capa_data():
    """Generates a mock CAPA log."""
    capas = [
        {"id": "CAPA-001", "event_id": "C-001", "status": "Closed - Effective", "due_date": datetime(2024, 5, 1), "owner": "Engineer A"},
        {"id": "CAPA-002", "event_id": "C-003", "status": "Pending VOE", "due_date": datetime(2024, 6, 20), "owner": "Engineer B"},
        {"id": "CAPA-003", "event_id": "C-005", "status": "Open", "due_date": datetime.now() + timedelta(days=10), "owner": "Engineer A"},
        {"id": "CAPA-004", "event_id": "C-002", "status": "Overdue", "due_date": datetime.now() - timedelta(days=5), "owner": "Engineer C"}
    ]
    for c in capas:
        c['due_date'] = pd.to_datetime(c['due_date'])
    return pd.DataFrame(capas)

# =================================================================================================
# MODULE: QTI COMMAND CENTER
# =================================================================================================
def show_command_center():
    st.title("🔬 QTI Command Center")
    st.markdown("A real-time overview of the quality system's health, active investigations, and process stability.")
    
    process_data = st.session_state['process_data']
    capa_data = st.session_state['capa_data']
    
    st.subheader("Key Performance Indicators (KPIs)")
    kpi_cols = st.columns(4)
    active_investigations = len(capa_data[capa_data['status'].isin(['Open', 'Pending VOE', 'Overdue'])])
    overdue_capas = len(capa_data[(capa_data['status'] != 'Closed - Effective') & (capa_data['due_date'] < datetime.now())])
    new_anomalies = len(process_data[process_data['timestamp'] > datetime.now() - timedelta(days=1)])
    mock_cpk = round(np.random.uniform(1.2, 1.5), 2)
    
    kpi_cols[0].metric("Active Investigations", active_investigations)
    kpi_cols[1].metric("Overdue CAPAs", overdue_capas, delta=overdue_capas, delta_color="inverse")
    kpi_cols[2].metric("New Data Points (24h)", new_anomalies)
    kpi_cols[3].metric("Avg. Process Cpk", mock_cpk, delta=round(mock_cpk-1.33, 2))

    st.markdown("---")
    
    col1, col2 = st.columns((2, 1.5))
    with col1:
        st.subheader("Process Line Health Status")
        line_health = process_data.groupby('line_id')['is_anomaly'].mean().reset_index()
        line_health = line_health.rename(columns={'is_anomaly': 'anomaly_rate'})
        fig = px.treemap(line_health, path=['line_id'], values='anomaly_rate', color='anomaly_rate', color_continuous_scale='RdYlGn_r', title="Process Line Anomaly Rate (Lower is Better)")
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This treemap visualizes the proportion of anomalous data points for each production line. Lines with higher rates (redder) require more immediate attention.")
    with col2:
        st.subheader("Active CAPA Queue")
        active_capas_df = capa_data[capa_data['status'] != 'Closed - Effective'].sort_values('due_date')
        st.dataframe(active_capas_df, use_container_width=True, hide_index=True)
        st.caption("This queue shows all non-closed CAPAs. Use this to track progress and prioritize follow-ups.")

# =================================================================================================
# MODULE: PROCESS MONITORING
# =================================================================================================
def show_process_monitoring():
    st.title("📈 Process Monitoring & Anomaly Detection")
    st.markdown("Use classical SPC and AI-driven methods to monitor process stability and detect deviations.")
    
    process_data = st.session_state['process_data']
    
    st.sidebar.header("Monitoring Filters")
    param_to_monitor = st.sidebar.selectbox("Select Parameter to Monitor:", ('reagent_ph', 'fill_volume_ml', 'pressure_psi'))
    line_to_monitor = st.sidebar.selectbox("Select Process Line:", sorted(process_data['line_id'].unique()))
    
    monitor_df = process_data[process_data['line_id'] == line_to_monitor].copy()
    monitor_df = monitor_df.sort_values('timestamp').reset_index(drop=True)
    
    tab1, tab2 = st.tabs(["Statistical Process Control (SPC)", "AI Anomaly Detection"])
    
    with tab1:
        st.subheader(f"SPC Control Chart for '{param_to_monitor}' on '{line_to_monitor}'")
        
        if not monitor_df.empty:
            # FIX: Replace incorrect 'pyspc.Spc' call with the correct functional call
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            chart_data = pyspc.i_mr_chart(monitor_df[param_to_monitor])
            axes[0].plot(chart_data['i'])
            axes[0].axhline(chart_data['i_cl'], color='green')
            axes[0].axhline(chart_data['i_ucl'], color='red', linestyle='--')
            axes[0].axhline(chart_data['i_lcl'], color='red', linestyle='--')
            axes[0].set_title(f'Individuals Chart for {param_to_monitor}')
            axes[1].plot(chart_data['mr'])
            axes[1].axhline(chart_data['mr_cl'], color='green')
            axes[1].axhline(chart_data['mr_ucl'], color='red', linestyle='--')
            axes[1].set_title('Moving Range Chart')
            st.pyplot(fig)
            
            st.write("Detected SPC Rule Violations:")
            violations = chart_data['i_violations']
            if not violations:
                st.success("No SPC rule violations detected. The process is in a state of statistical control.")
            else:
                st.warning("SPC rule violations detected! The process may be out of control.")
                st.write(violations)
        else:
            st.warning("No data available for the selected filters.")

    with tab2:
        st.subheader(f"AI-Powered Anomaly Detection for '{line_to_monitor}'")
        st.markdown("This method uses an Isolation Forest model to find 'unknown unknowns'—unusual data points that may not violate simple SPC rules but are abnormal in a multivariate context.")
        if not monitor_df.empty:
            features = ['reagent_ph', 'fill_volume_ml', 'pressure_psi']
            X = monitor_df[features]
            model = IsolationForest(contamination='auto', random_state=42)
            model.fit(X)
            monitor_df['anomaly_score'] = model.decision_function(X)
            monitor_df['is_flagged_anomaly'] = model.predict(X)
            fig = px.line(monitor_df, x='timestamp', y='anomaly_score', title='Anomaly Score Over Time (Lower is More Anomalous)', labels={'timestamp': 'Timestamp', 'anomaly_score': 'Anomaly Score'})
            anomalies = monitor_df[monitor_df['is_flagged_anomaly'] == -1]
            fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['anomaly_score'], mode='markers', name='Detected Anomaly', marker=dict(color='red', size=8)))
            st.plotly_chart(fig, use_container_width=True)
            st.write("Top 5 Most Anomalous Events Detected:")
            st.dataframe(anomalies.sort_values('anomaly_score').head(), use_container_width=True)

# =================================================================================================
# MODULE: EVENT TRIAGE
# =================================================================================================
@st.cache_resource
def get_nlp_model():
    """Trains and caches a simple NLP model for complaint classification."""
    complaint_df = generate_complaint_data()
    X = complaint_df['text']
    y = complaint_df['category']
    
    # FIX: Use sklearn.pipeline.Pipeline correctly
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(X, y)
    return pipeline

def show_event_triage():
    st.title(" triage Event Triage & Risk Prioritization")
    st.markdown("Log new events, use NLP to auto-classify complaints, and calculate risk to prioritize investigations.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Automated Complaint Classification (NLP)")
        nlp_model = get_nlp_model()
        new_complaint_text = st.text_area("Paste new complaint text here:", "The vials from the latest shipment were poorly sealed and the pressure was all over the place during our validation run.", height=150)
        if st.button("Classify Complaint"):
            prediction = nlp_model.predict([new_complaint_text])[0]
            st.success(f"Predicted Category: **{prediction}**")
            st.info("This classification helps route the event to the correct SME (e.g., a hardware engineer vs. a chemist).")
    with col2:
        st.subheader("Risk Priority Number (RPN) Calculator")
        st.markdown("Quantify risk using the FMEA methodology (RPN = Severity x Occurrence x Detection).")
        with st.form("rpn_form"):
            severity = st.slider("Severity (S): Impact on patient/customer safety", 1, 10, 5)
            occurrence = st.slider("Occurrence (O): Likelihood of the event happening", 1, 10, 3)
            detection = st.slider("Detection (D): How difficult is it to detect the failure?", 1, 10, 8)
            submitted = st.form_submit_button("Calculate RPN")
            if submitted:
                rpn = severity * occurrence * detection
                if rpn > 125: st.error(f"**Calculated RPN: {rpn}** (High Risk)")
                elif rpn > 60: st.warning(f"**Calculated RPN: {rpn}** (Medium Risk)")
                else: st.success(f"**Calculated RPN: {rpn}** (Low Risk)")
                st.caption("Use the RPN score to objectively prioritize which investigations to tackle first.")

# =================================================================================================
# MODULE: RCA WORKBENCH
# =================================================================================================
@st.cache_resource
def get_rca_model(_df):
    """Trains a RandomForest model for identifying feature importance in RCA."""
    features = ['reagent_ph', 'fill_volume_ml', 'pressure_psi']
    target = 'is_anomaly'
    X = _df[features]
    y = _df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def show_rca_workbench():
    st.title("🛠️ Root Cause Analysis (RCA) Workbench")
    st.markdown("A comprehensive toolkit to investigate events, test hypotheses, and identify the true root cause.")
    process_data = st.session_state['process_data']
    st.sidebar.header("RCA Filters")
    lot_to_investigate = st.sidebar.selectbox("Select Material Lot to Investigate:", sorted(process_data['material_lot'].unique()))
    investigation_df = process_data[process_data['material_lot'] == lot_to_investigate]
    tab1, tab2, tab3 = st.tabs(["Qualitative Analysis (Fishbone)", "Statistical Hypothesis Testing", "AI-Driven Root Cause"])
    with tab1:
        st.subheader("Fishbone (Ishikawa) Diagram")
        st.markdown("Brainstorm potential causes with your team and visualize them.")
        g = Digraph('G', filename='fishbone.gv')
        g.attr('node', shape='box')
        g.edge('Measurement', 'Effect')
        g.edge('Materials', 'Effect')
        g.edge('Personnel', 'Effect')
        g.edge('Environment', 'Effect')
        g.edge('Methods', 'Effect')
        g.edge('Machines', 'Effect')
        st.graphviz_chart(g)
        st.caption("In a full application, this would be an interactive whiteboard for team collaboration.")
    with tab2:
        st.subheader("Hypothesis Testing: Compare Two Groups")
        st.markdown("Use statistical tests to determine if observed differences are significant or just random chance.")
        param_to_test = st.selectbox("Select Parameter to Compare:", ('reagent_ph', 'pressure_psi'))
        group1 = process_data[process_data['material_lot'] == 'LOT-1'][param_to_test]
        group2 = process_data[process_data['material_lot'] == 'LOT-2'][param_to_test]
        fig = go.Figure()
        fig.add_trace(go.Box(y=group1, name='LOT-1'))
        fig.add_trace(go.Box(y=group2, name='LOT-2'))
        fig.update_layout(title=f'Comparison of {param_to_test} between Material Lots')
        st.plotly_chart(fig, use_container_width=True)
        ttest_res = stats.ttest_ind(group1, group2, equal_var=False)
        st.metric(label="T-test p-value", value=f"{ttest_res.pvalue:.4g}")
        if ttest_res.pvalue < 0.05: st.error("Result is statistically significant (p < 0.05). There is strong evidence of a real difference between the two lots.")
        else: st.success("Result is not statistically significant (p >= 0.05).")
    with tab3:
        st.subheader("AI-Driven Root Cause Identification")
        st.markdown("Use a Machine Learning model to rank the importance of different factors in causing anomalies.")
        rca_model = get_rca_model(process_data)
        features = ['reagent_ph', 'fill_volume_ml', 'pressure_psi']
        importances = pd.DataFrame({'feature': features, 'importance': rca_model.feature_importances_}).sort_values('importance', ascending=False)
        fig = px.bar(importances, x='feature', y='importance', title='Feature Importance for Anomaly Prediction')
        st.plotly_chart(fig, use_container_width=True)
        st.warning("**Interpretation:** The bar chart shows that **pressure_psi** and **reagent_ph** are the most predictive factors for anomalies in this dataset. This tells the engineer to focus their investigation on what could be causing variations in pressure and pH.")

# =================================================================================================
# MODULE: CAPA MANAGER
# =================================================================================================
def show_capa_manager():
    st.title("📋 CAPA Manager & Effectiveness Verification")
    st.markdown("Track the lifecycle of CAPAs and use data to prove that your solutions worked.")
    capa_data = st.session_state['capa_data']
    st.subheader("CAPA Log")
    st.dataframe(capa_data, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("Verification of Effectiveness (VOE) Analysis")
    capa_to_verify = st.selectbox("Select a CAPA to Verify:", capa_data['id'])
    if capa_to_verify:
        st.info("Demonstration: Simulating pre- and post-CAPA data for analysis.")
        pre_capa_data = np.random.normal(loc=10.2, scale=0.15, size=100) # Out of control process
        post_capa_data = np.random.normal(loc=10.0, scale=0.05, size=100) # Improved process
        voe_df = pd.DataFrame({'Measurement': np.concatenate([pre_capa_data, post_capa_data]), 'Phase': ['Pre-CAPA'] * 100 + ['Post-CAPA'] * 100})
        fig = px.box(voe_df, x='Phase', y='Measurement', color='Phase', title=f"Effectiveness of {capa_to_verify}: Pre- vs. Post-Implementation")
        st.plotly_chart(fig, use_container_width=True)
        voe_ttest = stats.ttest_ind(pre_capa_data, post_capa_data, equal_var=False)
        levene_test = stats.levene(pre_capa_data, post_capa_data)
        st.metric("p-value for Mean Improvement", f"{voe_ttest.pvalue:.4g}")
        st.metric("p-value for Variance Reduction (Levene's Test)", f"{levene_test.pvalue:.4g}")
        if voe_ttest.pvalue < 0.05 and levene_test.pvalue < 0.05: st.success("✅ **Effective:** The implemented CAPA resulted in a statistically significant improvement in both the process mean and its consistency (variance).")
        else: st.error("❌ **Not Effective:** The change did not result in a statistically significant improvement. Re-investigation is required.")

# =================================================================================================
# MODULE: PREDICTIVE ANALYTICS
# =================================================================================================
@st.cache_resource
def get_prediction_model_and_explainer(_df):
    """Trains a model and creates a SHAP explainer."""
    model = get_rca_model(_df)
    explainer = shap.TreeExplainer(model)
    return model, explainer

def show_predictive_analytics():
    st.title("🔮 Predictive Quality Analytics")
    st.markdown("Shift from reactive to proactive quality. Predict failures before they happen and optimize process parameters.")
    process_data = st.session_state['process_data']
    model, explainer = get_prediction_model_and_explainer(process_data)
    st.subheader("Real-Time Batch Failure Prediction")
    st.markdown("Adjust the process parameters below to see the predicted outcome and understand *why*.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Input Parameters:")
        ph_input = st.slider("Reagent pH", 7.0, 7.6, 7.25, 0.01)
        vol_input = st.slider("Fill Volume (mL)", 9.8, 10.2, 10.0, 0.01)
        psi_input = st.slider("Pressure (psi)", 45.0, 58.0, 51.0, 0.5)
        input_data = np.array([[ph_input, vol_input, psi_input]])
        prediction_proba = model.predict_proba(input_data)[0][1]
        st.metric("Predicted Probability of Anomaly", f"{prediction_proba:.1%}")
        if prediction_proba > 0.5: st.error("High risk of failure predicted for these settings.")
        else: st.success("Process settings are predicted to be stable.")
    with col2:
        st.write("eXplainable AI (XAI) Driver Analysis:")
        
        # FIX: Make SHAP value indexing robust
        shap_values_list = explainer.shap_values(input_data)
        # Check if the output has two arrays (for two classes) or just one
        if isinstance(shap_values_list, list) and len(shap_values_list) > 1:
            shap_values = shap_values_list[1] # Use values for the 'anomaly' class
            expected_value = explainer.expected_value[1]
        else:
            shap_values = shap_values_list # Use the only array available
            expected_value = explainer.expected_value
        
        fig, ax = plt.subplots()
        shap.force_plot(expected_value, shap_values, np.around(input_data.astype(float), 2), feature_names=['reagent_ph', 'fill_volume_ml', 'pressure_psi'], matplotlib=True, show=False)
        st.pyplot(fig, bbox_inches='tight', dpi=150)
        
        st.info("""**How to Read This Plot:**...""")

# =================================================================================================
# MAIN APP LOGIC
# =================================================================================================
def main():
    st.sidebar.title("QTI Workbench Navigation")
    st.sidebar.markdown("---")
    if 'process_data' not in st.session_state:
        st.session_state['process_data'] = generate_process_data()
        st.session_state['complaint_data'] = generate_complaint_data()
        st.session_state['capa_data'] = generate_capa_data()
    module = st.sidebar.radio("Select a Module:", ("QTI Command Center", "Process Monitoring", "Event Triage", "RCA Workbench", "CAPA Manager", "Predictive Analytics"))
    st.sidebar.markdown("---")
    st.sidebar.info("This application is a functional prototype demonstrating a comprehensive QTI workflow.")
    if module == "QTI Command Center": show_command_center()
    elif module == "Process Monitoring": show_process_monitoring()
    elif module == "Event Triage": show_event_triage()
    elif module == "RCA Workbench": show_rca_workbench()
    elif module == "CAPA Manager": show_capa_manager()
    elif module == "Predictive Analytics": show_predictive_analytics()

if __name__ == "__main__":
    main()
