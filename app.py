# =================================================================================================
# QTI ENGINEERING WORKBENCH - v1.0 Final
#
# AUTHOR: Subject Matter Expert AI
# DATE: 2024-07-23
#
# DESCRIPTION:
# This is a complete, single-file Streamlit application designed to serve as an end-to-end
# workbench for a Quality Technical Investigation (QTI) Engineer. It integrates methodologies
# from Quality Engineering, Data Science, and Operations to provide a comprehensive tool for
# process monitoring, event triage, root cause analysis, CAPA management, and predictive analytics.
#
# This file is self-contained and designed to be fully functional without modification.
# =================================================================================================

# --- 1. CORE LIBRARY IMPORTS ---
# For the core application framework, UI, and basic data handling.
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# --- 2. VISUALIZATION LIBRARY IMPORTS ---
# For creating interactive charts and diagrams.
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from graphviz import Digraph

# --- 3. ADVANCED ANALYTICS & ML LIBRARY IMPORTS ---
# For statistical analysis, machine learning models, and explainability.
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import scipy.stats as stats
import shap
# Import the 'spc' submodule correctly for object-oriented use
import pyspc.spc as pyspc_module


# =================================================================================================
# APPLICATION CONFIGURATION
# =================================================================================================

st.set_page_config(
    page_title="QTI Engineering Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =================================================================================================
# DATA SIMULATION LAYER
# This section simulates a realistic data environment. In a real-world deployment, these
# functions would be replaced with connectors to databases (e.g., LIMS, MES, QMS).
# =================================================================================================

@st.cache_data
def generate_process_data(num_records=2000):
    """
    Generates a realistic, multivariate process dataset.
    This data includes a systemic issue (higher pressure/pH in LOT-2) to make
    the RCA and predictive modules more meaningful.
    """
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=i*15) for i in range(num_records)]
    data = []
    for ts in timestamps:
        material_lot = f"LOT-{(ts.day % 2) + 1}"
        is_anomaly = 0
        base_ph, base_vol, base_psi = 7.2, 10.0, 50.0

        # Introduce a correlated, systemic issue for a specific material lot
        if material_lot == 'LOT-2':
            ph = base_ph + np.random.normal(0.1, 0.1)  # Shifted pH
            psi = base_psi + np.random.normal(2.5, 1.0) # Shifted Pressure
            if np.random.rand() > 0.85: # Higher probability of anomaly for this lot
                is_anomaly = 1
        else:  # LOT-1 is our "golden" lot
            ph = base_ph + np.random.normal(0, 0.05)
            psi = base_psi + np.random.normal(0, 0.5)
            if np.random.rand() > 0.98: # Lower baseline anomaly rate
                is_anomaly = 1
        
        vol = base_vol + np.random.normal(0, 0.03)

        record = {
            "timestamp": ts, "process_id": f"PROC-{(ts.day % 3) + 1}",
            "line_id": f"LINE-{(ts.hour % 4) + 1}", "reagent_ph": round(ph, 2),
            "fill_volume_ml": round(vol, 3), "pressure_psi": round(psi, 1),
            "operator_id": f"OP-{(ts.day % 5) + 1}",
            "batch_id": f"B-{ts.strftime('%Y%m%d')}-{((ts.hour // 8) % 3) + 1}",
            "material_lot": material_lot, "is_anomaly": is_anomaly
        }
        data.append(record)
    return pd.DataFrame(data)

@st.cache_data
def generate_complaint_data():
    """Generates a mock customer complaint log for NLP demonstrations."""
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
    """Generates a mock CAPA log for tracking and management."""
    capas = [
        {"id": "CAPA-001", "event_id": "C-001", "status": "Closed - Effective", "due_date": datetime(2024, 5, 1), "owner": "Engineer A"},
        {"id": "CAPA-002", "event_id": "C-003", "status": "Pending VOE", "due_date": datetime(2024, 6, 20), "owner": "Engineer B"},
        {"id": "CAPA-003", "event_id": "C-005", "status": "Open", "due_date": datetime.now() + timedelta(days=10), "owner": "Engineer A"},
        {"id": "CAPA-004", "event_id": "C-002", "status": "Overdue", "due_date": datetime.now() - timedelta(days=5), "owner": "Engineer C"}
    ]
    for c in capas: c['due_date'] = pd.to_datetime(c['due_date'])
    return pd.DataFrame(capas)


# =================================================================================================
# MODULE 1: QTI COMMAND CENTER
# =================================================================================================
def show_command_center():
    st.title("🔬 QTI Command Center")
    st.markdown("A real-time overview of the quality system's health, active investigations, and process stability.")
    process_data, capa_data = st.session_state['process_data'], st.session_state['capa_data']

    st.subheader("Key Performance Indicators (KPIs)")
    kpi_cols = st.columns(4)
    active_investigations = len(capa_data[capa_data['status'].isin(['Open', 'Pending VOE', 'Overdue'])])
    overdue_capas = len(capa_data[(capa_data['status'] != 'Closed - Effective') & (capa_data['due_date'] < datetime.now())])
    new_data_points = len(process_data[process_data['timestamp'] > datetime.now() - timedelta(days=1)])
    mock_cpk = round(np.random.uniform(1.2, 1.5), 2)
    kpi_cols[0].metric("Active Investigations", active_investigations)
    kpi_cols[1].metric("Overdue CAPAs", overdue_capas, delta=overdue_capas, delta_color="inverse")
    kpi_cols[2].metric("New Data Points (24h)", new_data_points)
    kpi_cols[3].metric("Avg. Process Cpk (Mock)", mock_cpk, delta=round(mock_cpk - 1.33, 2))

    st.markdown("---")
    col1, col2 = st.columns((2, 1.5))
    with col1:
        st.subheader("Process Line Health Status")
        line_health = process_data.groupby('line_id')['is_anomaly'].mean().reset_index().rename(columns={'is_anomaly': 'anomaly_rate'})
        fig = px.treemap(line_health, path=['line_id'], values='anomaly_rate', color='anomaly_rate', color_continuous_scale='RdYlGn_r', title="Process Line Anomaly Rate (Lower is Better)")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Active CAPA Queue")
        active_capas_df = capa_data[capa_data['status'] != 'Closed - Effective'].sort_values('due_date')
        st.dataframe(active_capas_df, use_container_width=True, hide_index=True)

# =================================================================================================
# MODULE 2: PROCESS MONITORING
# =================================================================================================
def show_process_monitoring():
    st.title("📈 Process Monitoring & Anomaly Detection")
    st.markdown("Use classical SPC and AI-driven methods to monitor process stability and detect deviations.")
    process_data = st.session_state['process_data']
    st.sidebar.header("Monitoring Filters")
    param_to_monitor = st.sidebar.selectbox("Select Parameter to Monitor:", ('reagent_ph', 'fill_volume_ml', 'pressure_psi'))
    line_to_monitor = st.sidebar.selectbox("Select Process Line:", sorted(process_data['line_id'].unique()))
    monitor_df = process_data[process_data['line_id'] == line_to_monitor].copy().sort_values('timestamp').reset_index(drop=True)
    
    tab1, tab2 = st.tabs(["Statistical Process Control (SPC)", "AI Anomaly Detection"])
    with tab1:
        st.subheader(f"SPC Control Chart for '{param_to_monitor}' on '{line_to_monitor}'")
        if not monitor_df.empty:
            # CORRECTED: Instantiate the Spc object from the imported submodule.
            spc_chart = pyspc_module.Spc(monitor_df[param_to_monitor], chart_type='i', title=f'I-MR Chart for {param_to_monitor}')
            fig = spc_chart.get_fig()
            st.pyplot(fig)
            plt.close(fig) # Best practice to close the matplotlib figure after use.
            violations = spc_chart.get_violating_points()
            st.write("Detected SPC Rule Violations:")
            if violations.empty:
                st.success("No SPC rule violations detected. The process appears to be in a state of statistical control.")
            else:
                st.warning("SPC rule violations detected! The process may be out of control.")
                st.dataframe(violations, use_container_width=True)
    with tab2:
        st.subheader(f"AI-Powered Anomaly Detection for '{line_to_monitor}'")
        st.markdown("This method uses an Isolation Forest model to find 'unknown unknowns'—unusual data points that may not violate simple SPC rules but are abnormal in a multivariate context.")
        if not monitor_df.empty:
            features = ['reagent_ph', 'fill_volume_ml', 'pressure_psi']
            X = monitor_df[features]
            model = IsolationForest(contamination='auto', random_state=42).fit(X)
            monitor_df['anomaly_score'] = model.decision_function(X)
            monitor_df['is_flagged_anomaly'] = model.predict(X)
            fig = px.line(monitor_df, x='timestamp', y='anomaly_score', title='Anomaly Score Over Time (Lower is More Anomalous)')
            anomalies = monitor_df[monitor_df['is_flagged_anomaly'] == -1]
            fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['anomaly_score'], mode='markers', name='Detected Anomaly', marker=dict(color='red', size=8)))
            st.plotly_chart(fig, use_container_width=True)
            st.write("Top 5 Most Anomalous Events Detected:")
            st.dataframe(anomalies.sort_values('anomaly_score').head(), use_container_width=True)

# =================================================================================================
# MODULE 3: EVENT TRIAGE
# =================================================================================================
@st.cache_resource
def get_nlp_model():
    """Trains and caches a simple NLP model for complaint classification."""
    complaint_df = generate_complaint_data()
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
    pipeline.fit(complaint_df['text'], complaint_df['category'])
    return pipeline

def show_event_triage():
    st.title("📥 Event Triage & Risk Prioritization")
    st.markdown("Log new events, use NLP to auto-classify complaints, and calculate risk to prioritize investigations.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Automated Complaint Classification (NLP)")
        nlp_model = get_nlp_model()
        new_complaint_text = st.text_area("Paste new complaint text here:", "The vials from the latest shipment were poorly sealed and the pressure was all over the place during our validation run.", height=150)
        if st.button("Classify Complaint"):
            prediction = nlp_model.predict([new_complaint_text])[0]
            st.success(f"Predicted Category: **{prediction}**")
            st.info("This classification helps route the event to the correct SME.")
    with col2:
        st.subheader("Risk Priority Number (RPN) Calculator")
        st.markdown("Quantify risk using the FMEA methodology (RPN = Severity x Occurrence x Detection).")
        with st.form("rpn_form"):
            severity = st.slider("Severity (S): Impact on patient/customer safety", 1, 10, 5)
            occurrence = st.slider("Occurrence (O): Likelihood of the event happening", 1, 10, 3)
            detection = st.slider("Detection (D): How difficult is it to detect the failure?", 1, 10, 8)
            if st.form_submit_button("Calculate RPN"):
                rpn = severity * occurrence * detection
                if rpn > 125: st.error(f"**Calculated RPN: {rpn}** (High Risk - Immediate Action Required)")
                elif rpn > 60: st.warning(f"**Calculated RPN: {rpn}** (Medium Risk - Investigation Warranted)")
                else: st.success(f"**Calculated RPN: {rpn}** (Low Risk - Monitor)")

# =================================================================================================
# MODULE 4: RCA WORKBENCH
# =================================================================================================
@st.cache_resource
def get_rca_model(_df):
    """Trains a RandomForest model for identifying feature importance in RCA."""
    features, target = ['reagent_ph', 'fill_volume_ml', 'pressure_psi'], 'is_anomaly'
    X, y = _df[features], _df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train, y_train)

def show_rca_workbench():
    st.title("🛠️ Root Cause Analysis (RCA) Workbench")
    st.markdown("A comprehensive toolkit to investigate events, test hypotheses, and identify the true root cause.")
    process_data = st.session_state['process_data']
    st.sidebar.header("RCA Filters")
    lot_to_investigate = st.sidebar.selectbox("Select Material Lot to Investigate:", sorted(process_data['material_lot'].unique()))
    
    tab1, tab2, tab3 = st.tabs(["Qualitative Analysis (Fishbone)", "Statistical Hypothesis Testing", "AI-Driven Root Cause"])
    with tab1:
        st.subheader("Fishbone (Ishikawa) Diagram")
        st.markdown("Use this diagram to brainstorm potential causes with your team and visualize them.")
        g = Digraph('G'); g.attr('node', shape='box')
        for cat in ['Measurement', 'Materials', 'Personnel', 'Environment', 'Methods', 'Machines']: g.edge(cat, 'Effect')
        st.graphviz_chart(g)
    with tab2:
        st.subheader("Hypothesis Testing: Compare Two Groups")
        st.markdown("Use statistical tests to determine if observed differences are significant or just random chance.")
        param_to_test = st.selectbox("Select Parameter to Compare:", ('reagent_ph', 'pressure_psi'))
        group1, group2 = process_data[process_data['material_lot'] == 'LOT-1'][param_to_test], process_data[process_data['material_lot'] == 'LOT-2'][param_to_test]
        fig = go.Figure(); fig.add_trace(go.Box(y=group1, name='LOT-1')); fig.add_trace(go.Box(y=group2, name='LOT-2'))
        fig.update_layout(title=f'Comparison of {param_to_test} between Material Lots'); st.plotly_chart(fig, use_container_width=True)
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
        st.warning("**Interpretation:** The bar chart ranks process variables by their power in predicting an anomaly. The engineer should focus their investigation on the top-ranked factors—in this case, pressure and pH.")

# =================================================================================================
# MODULE 5: CAPA MANAGER
# =================================================================================================
def show_capa_manager():
    st.title("📋 CAPA Manager & Effectiveness Verification")
    st.markdown("Track the lifecycle of CAPAs and use data to prove that your solutions worked.")
    capa_data = st.session_state['capa_data']
    st.subheader("CAPA Log"); st.dataframe(capa_data, use_container_width=True, hide_index=True)
    st.markdown("---"); st.subheader("Verification of Effectiveness (VOE) Analysis")
    capa_to_verify = st.selectbox("Select a CAPA to Verify:", capa_data['id'])
    if capa_to_verify:
        st.info("Demonstration: Simulating pre- and post-CAPA data for analysis.")
        pre_capa_data, post_capa_data = np.random.normal(10.2, 0.15, 100), np.random.normal(10.0, 0.05, 100)
        voe_df = pd.DataFrame({'Measurement': np.concatenate([pre_capa_data, post_capa_data]), 'Phase': ['Pre-CAPA'] * 100 + ['Post-CAPA'] * 100})
        fig = px.box(voe_df, x='Phase', y='Measurement', color='Phase', title=f"Effectiveness of {capa_to_verify}: Pre- vs. Post-Implementation")
        st.plotly_chart(fig, use_container_width=True)
        ttest, levene = stats.ttest_ind(pre_capa_data, post_capa_data, equal_var=False), stats.levene(pre_capa_data, post_capa_data)
        st.metric("p-value for Mean Improvement", f"{ttest.pvalue:.4g}"); st.metric("p-value for Variance Reduction (Levene's Test)", f"{levene.pvalue:.4g}")
        if ttest.pvalue < 0.05 and levene.pvalue < 0.05: st.success("✅ **Effective:** The implemented CAPA resulted in a statistically significant improvement in both the process mean and its consistency (variance).")
        else: st.error("❌ **Not Effective:** No significant improvement detected. Re-investigation is required.")

# =================================================================================================
# MODULE 6: PREDICTIVE ANALYTICS
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
        ph_input, vol_input, psi_input = st.slider("Reagent pH", 7.0, 7.6, 7.25, 0.01), st.slider("Fill Volume (mL)", 9.8, 10.2, 10.0, 0.01), st.slider("Pressure (psi)", 45.0, 58.0, 51.0, 0.5)
        input_data = np.array([[ph_input, vol_input, psi_input]])
        prediction_proba = model.predict_proba(input_data)[0][1]
        st.metric("Predicted Probability of Anomaly", f"{prediction_proba:.1%}")
        if prediction_proba > 0.5: st.error("High risk of failure predicted for these settings.")
        else: st.success("Process settings are predicted to be stable.")
    with col2:
        st.write("eXplainable AI (XAI) Driver Analysis:")
        # CORRECTED: Robustly call shap.force_plot with the correct arguments and data shapes.
        base_value = explainer.expected_value[1]
        shap_values_output = explainer.shap_values(input_data)[1] # Get SHAP values for class 1 (anomaly)
        
        # The function requires a 1D array for a single prediction.
        if shap_values_output.ndim > 1:
            shap_values_for_plot = shap_values_output[0]
        else:
            shap_values_for_plot = shap_values_output

        fig, ax = plt.subplots(figsize=(10, 3))
        shap.force_plot(base_value, shap_values_for_plot, np.around(input_data[0], 2), feature_names=['reagent_ph', 'fill_volume_ml', 'pressure_psi'], matplotlib=True, show=False)
        st.pyplot(fig, bbox_inches='tight', dpi=150)
        plt.close(fig) # Close the figure to prevent memory leaks

# =================================================================================================
# MAIN APPLICATION LOGIC
# =================================================================================================
def main():
    """Main function to define the app's navigation and structure."""
    st.sidebar.title("QTI Workbench Navigation")
    st.sidebar.markdown("---")
    # Initialize session state for data persistence across "pages"
    if 'process_data' not in st.session_state:
        st.session_state['process_data'] = generate_process_data()
        st.session_state['complaint_data'] = generate_complaint_data()
        st.session_state['capa_data'] = generate_capa_data()
    
    # Use a dictionary for clean page routing
    page_functions = {
        "QTI Command Center": show_command_center,
        "Process Monitoring": show_process_monitoring,
        "Event Triage": show_event_triage,
        "RCA Workbench": show_rca_workbench,
        "CAPA Manager": show_capa_manager,
        "Predictive Analytics": show_predictive_analytics,
    }
    module = st.sidebar.radio("Select a Module:", tuple(page_functions.keys()))
    st.sidebar.markdown("---")
    st.sidebar.info("This application is a functional prototype demonstrating a comprehensive QTI workflow.")
    
    # Call the selected page's function
    page_functions[module]()

if __name__ == "__main__":
    main()
