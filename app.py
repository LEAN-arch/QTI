# app.py (Corrected)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# --- Page Configuration ---
st.set_page_config(
    page_title="Quality Engineer Application | [Your Name]",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions & Advanced Data Generation ---
@st.cache_data
def create_advanced_mock_data(issue=False, num_batches=100):
    """Creates more realistic process data for advanced analysis."""
    np.random.seed(42)
    
    # Baseline process
    base_mean = 10.0
    base_std = 0.08
    
    if issue:
        # Introduce a process shift and increased variability
        pre_shift = np.random.normal(loc=base_mean, scale=base_std, size=int(num_batches/2))
        post_shift_mean = 10.15
        post_shift_std = 0.15
        post_shift = np.random.normal(loc=post_shift_mean, scale=post_shift_std, size=int(num_batches/2))
        data = np.concatenate([pre_shift, post_shift])
    else:
        # A stable, improved process
        data = np.random.normal(loc=base_mean, scale=base_std, size=num_batches)

    df = pd.DataFrame({'Batch': range(1, num_batches + 1), 'Measurement': data})
    
    # Add potential root cause factors
    operators = ['Operator A', 'Operator B', 'Operator C']
    machines = ['Machine 1', 'Machine 2']
    df['Operator'] = np.random.choice(operators, num_batches)
    df['Machine'] = np.random.choice(machines, num_batches)
    
    # Simulate a correlation for the 'issue' scenario
    if issue:
        # Make Operator C's results trend higher, aligning with the shift
        mask = (df['Batch'] > num_batches/2)
        df.loc[mask, 'Operator'] = np.random.choice(['Operator B', 'Operator C'], size=mask.sum(), p=[0.3, 0.7])
        df.loc[mask, 'Measurement'] += np.where(df.loc[mask, 'Operator'] == 'Operator C', 0.1, -0.05)
        
    return df

@st.cache_data
def get_pareto_data():
    """Generates mock defect data for Pareto analysis."""
    return pd.DataFrame([
        {'Defect Type': 'Incorrect Reagent Concentration', 'Count': 88},
        {'Defect Type': 'Mislabeled Vials', 'Count': 35},
        {'Defect Type': 'Leaking Caps', 'Count': 12},
        {'Defect Type': 'Cosmetic Blemish', 'Count': 6},
        {'Defect Type': 'Documentation Error', 'Count': 4},
    ]).sort_values(by='Count', ascending=False)

def calculate_capability(data, usl, lsl):
    """Calculates Cpk and Ppk."""
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1) # Sample standard deviation
    
    if std_dev == 0: return 0, 0
    
    cpu = (usl - mean) / (3 * std_dev)
    cpl = (mean - lsl) / (3 * std_dev)
    cpk = min(cpu, cpl)
    
    pp = (usl - lsl) / (6 * std_dev)
    ppk = min((usl - mean) / (3 * std_dev), (mean - lsl) / (3 * std_dev))
    
    return cpk, ppk

# --- Main App Structure ---
def run_app():
    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### [Your Name]")
    st.sidebar.markdown("**Applying for: Quality Engineer, QTI**")
    st.sidebar.markdown("📍 San Diego, CA")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Explore My Qualifications",
        ["Welcome & Introduction", "Core: Investigations", "Core: Leadership & Collaboration", "Core: Continuous Improvement", "Regulatory & IVD Expertise", "Qualifications & Contact"]
    )

    # Dictionary-based routing
    pages = {
        "Welcome & Introduction": show_introduction,
        "Core: Investigations": show_investigations_module,
        "Core: Leadership & Collaboration": show_leadership_module,
        "Core: Continuous Improvement": show_continuous_improvement_module,
        "Regulatory & IVD Expertise": show_regulatory_module,
        "Qualifications & Contact": show_contact_module
    }
    pages[page]()

# --- Page Rendering Functions ---

def show_introduction():
    st.title("🔬 Interactive Application for the Quality Engineer (QTI) Role")
    st.markdown("---")
    st.subheader("To the Hiring Manager and the Diagnostics QA Team,")
    st.markdown("""
    Welcome. This interactive application is designed to demonstrate my qualifications for the **Quality Engineer** role within your **Quality Technical Investigation (QTI) team**. Having analyzed your company's focus and the specific requirements of this position, I've tailored this dashboard to go beyond a traditional resume.
    
    Here, you will find interactive simulations, case studies from my portfolio, and detailed explanations of how my experience in **investigations, leadership, and continuous improvement** directly aligns with your needs. My work is data-driven, deeply rooted in regulatory compliance, and focused on the Diagnostics and Life Sciences industry here in **San Diego**.
    
    Please use the sidebar to navigate through the modules. Each section is designed to provide concrete evidence of my ability to meet and exceed the expectations outlined in the job description.
    """)
    st.info("💡 **Tip:** This entire application is a single, self-contained Streamlit app, demonstrating my ability to create clear, data-centric communication tools for technical and non-technical audiences.")

def show_investigations_module():
    st.header("Core Competency: Investigations (CAPA, NCEs, Complaints)")
    st.markdown("This module demonstrates my systematic approach to the entire investigation lifecycle, using advanced statistical tools to drive from issue identification to robust, data-proven verification of effectiveness.")

    tab1, tab2, tab3 = st.tabs(["**1. Investigation Lifecycle & SPC**", "**2. Advanced RCA Toolkit**", "**3. Documentation & Reporting**"])

    # Shared data across tabs
    issue_data = create_advanced_mock_data(issue=True)
    lsl, target, usl = 9.7, 10.0, 10.3

    with tab1:
        st.subheader("Phase 1: Issue Identification with Statistical Process Control (SPC)")
        st.markdown("""
        The first step in any investigation is the robust detection of a quality issue. SPC is the primary tool for this. Below is an Individuals and Moving Range (I-MR) chart, ideal for monitoring individual batch data over time.
        """)

        # I-Chart (Individuals Chart)
        fig_i = px.line(issue_data, x='Batch', y='Measurement', title="I-Chart: Reagent Concentration (mM)", markers=True)
        fig_i.add_hline(y=target, line_dash="dot", line_color="green", annotation_text="Target")
        fig_i.add_hline(y=usl, line_dash="dash", line_color="red", annotation_text="UCL (3σ)")
        fig_i.add_hline(y=lsl, line_dash="dash", line_color="red", annotation_text="LCL (3σ)")
        fig_i.add_vrect(x0=50, x1=100, fillcolor="red", opacity=0.15, line_width=0, annotation_text="Process Shift", annotation_position="top left")
        st.plotly_chart(fig_i, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            **Mathematical Basis:**
            - **Centerline:** The process average (μ).
            - **Control Limits (UCL/LCL):** Calculated as μ ± 3σ, where σ is the process standard deviation, estimated from the moving range. These represent the expected range of common-cause variation.
            
            **Regulatory Basis:**
            - **21 CFR 820.100(a):** Requires "analyzing... process... data to identify existing and potential causes of nonconforming product." SPC charts are a primary method for fulfilling this requirement.
            - **ISO 13485:2016, 8.4:** Mandates the "analysis of data" to demonstrate QMS suitability and identify improvement opportunities.
            """)
        with col2:
            st.error("**Interpretation & Action:**")
            st.markdown("""
            - **Observation:** Around Batch 50, the process shows a clear upward shift. Multiple points are trending above the centerline and several points are violating the UCL.
            - **Conclusion:** The process is out of statistical control due to a "special cause" variation.
            - **Action:** This is objective evidence to trigger a Non-Conformance Event (NCE) and initiate a formal investigation and root cause analysis.
            """)
            
        st.markdown("---")
        st.subheader("Phase 4: Verification of Effectiveness (VOE) with Process Capability")
        st.markdown("After implementing a CAPA (e.g., reverting to a qualified raw material), we must prove the solution was effective. A control chart shows stability, but a **Process Capability Analysis** quantifies the process's ability to meet specifications.")

        if st.toggle("Simulate Implemented CAPA & Run VOE Analysis", value=False):
            voe_data_df = create_advanced_mock_data(issue=False)
            voe_data = voe_data_df['Measurement']
            
            cpk, ppk = calculate_capability(voe_data, usl, lsl)
            
            fig_hist = px.histogram(voe_data, nbins=20, title="Process Capability Histogram (Post-CAPA)")
            fig_hist.add_vline(x=target, line_dash="dot", line_color="green", annotation_text="Target")
            fig_hist.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
            fig_hist.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
            fig_hist.add_vline(x=voe_data.mean(), line_dash="solid", line_color="orange", annotation_text="Process Mean")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Process Capability Index (Cpk)", value=f"{cpk:.2f}")
                # FIX: Used raw string literal (r"...") to prevent SyntaxWarning
                st.markdown(r"""
                **Mathematical Basis (Cpk):**
                - $C_{pk} = min(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma})$
                - Measures how centered the process is and how well it fits within specification limits. It assesses potential (short-term) capability.
                """)
                st.info("**Industry Benchmark:** A Cpk > 1.33 is generally considered 'capable' for most processes.")
            with col2:
                st.success("**Interpretation & Action:**")
                st.markdown(f"""
                - **Observation:** The post-CAPA process is centered near the target and fits comfortably within the specification limits. The calculated Cpk of **{cpk:.2f}** exceeds the 1.33 benchmark.
                - **Conclusion:** The corrective action was effective. We have statistically proven that the process is not only stable but also highly capable of meeting quality requirements.
                - **Action:** Close the CAPA investigation. Continue monitoring the process via SPC as part of the Control Plan.
                """)

    with tab2:
        st.subheader("Advanced Toolkit for Efficient Root-Cause Determination")
        st.markdown("A single tool is never enough. I apply a suite of analytical methods to efficiently narrow down and confirm the root cause with cross-functional teams.")
        
        st.markdown("#### Tool 1: Pareto Analysis (Focusing the Effort)")
        st.markdown("""
        When faced with multiple defect types, Pareto analysis helps prioritize by focusing on the "vital few" causes that create the most problems (the 80/20 rule). This is a critical first step to ensure investigation resources are used effectively.
        """)
        pareto_df = get_pareto_data()
        pareto_df['Cumulative Percentage'] = (pareto_df['Count'].cumsum() / pareto_df['Count'].sum()) * 100

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=pareto_df['Defect Type'], y=pareto_df['Count'], name='Defect Count'))
        fig_pareto.add_trace(go.Scatter(x=pareto_df['Defect Type'], y=pareto_df['Cumulative Percentage'], name='Cumulative %', yaxis='y2', mode='lines+markers'))
        fig_pareto.update_layout(
            title="Pareto Chart of Production Defects",
            yaxis_title="Count",
            yaxis2=dict(title="Cumulative Percentage", overlaying='y', side='right', range=[0, 105]),
            legend=dict(x=0.7, y=0.9)
        )
        st.plotly_chart(fig_pareto, use_container_width=True)
        st.warning("**Interpretation & Action:**", icon="🎯")
        st.markdown("""
        - **Observation:** 'Incorrect Reagent Concentration' and 'Mislabeled Vials' account for over 80% of all defects.
        - **Action:** The investigation should immediately focus on the processes related to reagent formulation and vial labeling. We de-prioritize cosmetic issues for now. This directly informs the scope of the investigation.
        """)

        st.markdown("---")
        st.markdown("#### Tool 2: Hypothesis Testing (Confirming the Cause)")
        st.markdown("""
        Once potential causes are brainstormed (e.g., via a Fishbone diagram), we use statistical tests to find objective evidence. Here, we test the hypothesis that different operators contribute differently to the measurement variation.
        """)
        
        fig_box = px.box(issue_data, x='Operator', y='Measurement', color='Operator', title='Reagent Concentration by Operator')
        st.plotly_chart(fig_box, use_container_width=True)
        
        # ANOVA Test
        f_val, p_val = stats.f_oneway(
            issue_data[issue_data['Operator'] == 'Operator A']['Measurement'],
            issue_data[issue_data['Operator'] == 'Operator B']['Measurement'],
            issue_data[issue_data['Operator'] == 'Operator C']['Measurement']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Statistical Test:** One-Way ANOVA (Analysis of Variance)
            - **Null Hypothesis (H₀):** The mean measurement is the same across all operators.
            - **Alternative Hypothesis (Hₐ):** At least one operator's mean measurement is different.
            - **P-value:** The probability of observing the data if the null hypothesis is true. A low p-value (< 0.05) suggests we reject H₀.
            """)
            st.metric(label="ANOVA P-value", value=f"{p_val:.4f}")
        with col2:
            st.error("**Interpretation & Action:**", icon="🔬")
            st.markdown(f"""
            - **Observation:** The box plot shows that Operator C's results are visibly higher than the others. The extremely low p-value ({p_val:.4f}) provides strong statistical evidence that this difference is not due to random chance.
            - **Conclusion:** We have confirmed that 'Operator' is a significant factor in the process shift.
            - **Action:** The root cause investigation now focuses on *why* Operator C's results are different. Is it a training issue? A procedural deviation? A specific time of day? This leads to a more targeted and effective CAPA.
            """)

    with tab3:
        st.subheader("Maintaining Accurate, Clear, and Auditable Records")
        st.markdown("""
        Detailed and clear documentation is the foundation of a compliant Quality System. Investigation reports must be robust enough for internal review, management reporting, and regulatory audits (e.g., by the FDA or a Notified Body). My approach ensures reports are data-driven and directly reference objective evidence.
        """)
        st.text_area(
        "Sample Investigation Report Outline (with Objective Evidence links)",
        """
        **1.0 Executive Summary:**
           - Issue, Root Cause, CAPA Plan, Effectiveness Summary.
        **2.0 Investigation Details:**
           - 2.1 Description of Non-conformance (Ref: I-Chart in Section 7.1)
           - 2.2 Scope & Product Impact Assessment (Risk Assessment per ISO 14971)
           - 2.3 Immediate Corrections / Containment Actions
        **3.0 Root Cause Analysis:**
           - 3.1 Investigation Team (Cross-functional: QA, OPS, R&D)
           - 3.2 Data & Evidence Collected:
               - Pareto Analysis (Ref: Section 7.2)
               - ANOVA Test Results (Ref: Section 7.3)
           - 3.3 RCA Methodology (e.g., 5 Whys informed by data)
           - 3.4 Confirmed Root Cause(s): Confirmed procedural gap for Operator C.
        **4.0 Corrective and Preventive Action (CAPA) Plan:**
           - 4.1 Corrective Actions (e.g., Retrain Operator C)
           - 4.2 Preventive Actions (e.g., Update SOP and retrain all operators)
           - 4.3 Action Owners and Due Dates
        **5.0 Verification of Effectiveness (VOE):**
           - 5.1 VOE Plan and Acceptance Criteria (e.g., Cpk > 1.33 for 30 consecutive batches)
           - 5.2 VOE Data and Results Analysis (Ref: Process Capability Chart in Section 7.4)
           - 5.3 Conclusion of Effectiveness
        **6.0 Appendices & Signatures**
           - 7.1 I-Chart showing process shift
           - 7.2 Pareto Chart of related defects
           - 7.3 Box Plot and ANOVA output
           - 7.4 VOE Process Capability Report
        """, height=400)
        st.info("**Regulatory Basis:** This structured, evidence-based approach directly satisfies **21 CFR 820.198 (Complaint Files)** and **820.100 (CAPA)**, which require that all investigation activities are documented, including the statistical methods used.")

def show_leadership_module():
    st.header("Core Competency: Leadership & Collaboration")
    st.markdown("Effective quality engineering requires leading cross-functional teams and communicating clearly at all levels. This section highlights my experience in project management and stakeholder alignment.")

    tab1, tab2 = st.tabs(["**1. Cross-Functional Project Command Center**", "**2. Stakeholder Communication Examples**"])

    with tab1:
        st.subheader("Leading Projects with Ownership")
        st.write("I apply project management principles to plan, execute, and close investigations in a cross-functional environment. Below are examples from my portfolio, framed as project summaries.")

        with st.expander("Project Example: sPMO for a Global IVD Company (Werfen)"):
            st.markdown("""
            - **Objective:** Establish a strategic PMO command center to track and manage a portfolio of diagnostic product development projects.
            - **My Role:** Led the design of the dashboard logic, defined key performance indicators (KPIs) with stakeholders, and translated complex project data into actionable executive insights.
            - **Cross-Functional Teams Involved:** R&D, Regulatory Affairs, Manufacturing, and Executive Leadership.
            - **Outcome:** A centralized, real-time view of the entire project portfolio, enabling proactive risk management and resource allocation.
            - **Portfolio Link:** [sPMO Command Center](https://pmocommandcenter.streamlit.app/)
            """)
        
        with st.expander("Project Example: Tech Transfer & V&V Management (Grifols)"):
            st.markdown("""
            - **Objective:** Develop a tool to manage the complex Verification & Validation (V&V) activities during a technology transfer process.
            - **My Role:** Collaborated with MSAT and QA teams to map the V&V workflow, identify critical milestones, and create a system for tracking protocol execution, deviations, and report approvals.
            - **Cross-Functional Teams Involved:** MSAT (Manufacturing, Science, and Technology), QC, QA, Process Engineering.
            - **Outcome:** Reduced documentation errors and improved on-time completion of V&V activities by providing clear visibility into task status and dependencies.
            - **Portfolio Link:** [V&V Transfer Management](https://grifols-vvtransfer-management.streamlit.app/)
            """)

    with tab2:
        st.subheader("Communicating Effectively to All Levels")
        st.write("I tailor my communication to the audience, from technical deep-dives with SMEs to high-level summaries for leadership.")

        st.markdown("##### For Executive Leadership:")
        st.write("Clear, concise dashboards focusing on KPIs, risk, and strategic impact. This example shows high-level CAPA metrics.")
        kpi_data = {'Metric': ['Open CAPAs', 'Overdue CAPAs', 'Avg. Cycle Time (Days)', 'Effectiveness Check Pending'],
                    'Value': [25, 3, 62, 8]}
        kpi_df = pd.DataFrame(kpi_data)
        fig_kpi = go.Figure(go.Table(
            header=dict(values=list(kpi_df.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[kpi_df.Metric, kpi_df.Value], fill_color='lavender', align='left')))
        fig_kpi.update_layout(title_text="Executive CAPA Status Summary")
        st.plotly_chart(fig_kpi, use_container_width=True)

        st.markdown("##### For Technical Teams (R&D, OPS, QC):")
        st.write("Detailed process data, control charts, and statistical outputs for root cause analysis.")
        st.image("https://i.imgur.com/eN7G96t.png", caption="Example Technical Dashboard: I-Chart and Process Capability Analysis provided to SMEs for deep-dive investigation.")
        
def show_continuous_improvement_module():
    st.header("Core Competency: Continuous Improvement")
    st.markdown("I am committed to a culture of continuous improvement, using data to identify risks, gaps, and opportunities that drive sustainable value for the business.")

    tab1, tab2 = st.tabs(["**1. Quantifying Opportunity (COPQ)**", "**2. Proactive Process Analysis**"])

    with tab1:
        st.subheader("Justifying Improvement with Cost of Poor Quality (COPQ)")
        st.markdown("""
        Before launching a resource-intensive improvement project (like a Six Sigma DMAIC project), it's critical to build a business case. The Cost of Poor Quality (COPQ) framework translates defects, rework, and complaints into a financial metric that leadership can understand and act upon.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Internal Failure Costs")
            rework = st.number_input("Rework & Re-testing ($/yr)", value=25000, step=1000)
            scrap = st.number_input("Scrapped Materials ($/yr)", value=30000, step=1000)
            
            st.markdown("#### External Failure Costs")
            complaints = st.number_input("Complaint Investigations ($/yr)", value=15000, step=1000)
            warranty = st.number_input("Warranty/Returns ($/yr)", value=5000, step=1000)

        with col2:
            st.markdown("#### Appraisal & Prevention Costs")
            inspection = st.number_input("Inspection & Testing ($/yr)", value=75000, step=1000)
            prevention = st.number_input("QMS & Training ($/yr)", value=40000, step=1000)
            
            total_copq = rework + scrap + complaints + warranty + inspection + prevention
            st.metric(label="Total Estimated Cost of Poor Quality (COPQ)", value=f"${total_copq:,}")

        st.info("**Interpretation & Action:**", icon="💡")
        st.markdown(f"""
        - A COPQ of **${total_copq:,}** provides a powerful financial justification for dedicating resources to a Six Sigma or other continuous improvement project. 
        - **Action:** Present this analysis to stakeholders to gain approval for a project aimed at reducing the top drivers of this cost (e.g., Rework and Scrap), directly linking quality efforts to business value. This aligns with the principles of **ISO 13485:2016, 5.1.1 (Leadership and Commitment)**, which requires top management to ensure the QMS is effective.
        """)

    with tab2:
        st.subheader("Proactively Identifying Relationships with Correlation Analysis")
        st.markdown("""
        Continuous improvement isn't just about reacting to problems; it's about proactively understanding a process to prevent issues. Correlation analysis helps identify which input parameters have the strongest relationship with a critical quality output, guiding future control and optimization efforts.
        """)
        
        # Create correlated data for demonstration
        np.random.seed(10)
        size = 100
        temp = np.random.normal(70, 5, size)
        pressure = np.random.normal(30, 2, size)
        # Create a strong correlation between purity and temp, with some noise
        purity = 100 - (temp - 68) * 0.5 + np.random.normal(0, 0.5, size)
        corr_df = pd.DataFrame({'Temperature (C)': temp, 'Pressure (psi)': pressure, 'Product Purity (%)': purity})
        
        fig_corr = px.imshow(corr_df.corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto", title="Process Parameter Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Mathematical Basis:**
            - **Pearson Correlation Coefficient:** Measures the linear relationship between two variables.
            - **+1:** Perfect positive correlation.
            - **-1:** Perfect negative correlation.
            - **0:** No linear correlation.
            
            **Robustness:**
            - This analysis is a starting point. It requires sufficient, reliable process data. It shows association, not causation.
            """)
        with col2:
            st.warning("**Interpretation & Action:**", icon="📈")
            st.markdown("""
            - **Observation:** There is a strong negative correlation (-0.95) between Temperature and Product Purity. As temperature increases, purity decreases. Pressure shows a very weak correlation.
            - **Conclusion:** Temperature is a critical process parameter that must be tightly controlled.
            - **Action:**
                1.  Implement tighter SPC controls on Temperature.
                2.  If further optimization is needed, focus Design of Experiments (DOE) on the Temperature parameter, saving time and resources by not investigating Pressure. This is a key part of the "Analyze" and "Improve" phases of DMAIC.
            """)
        
def show_regulatory_module():
    st.header("Regulatory & In Vitro Diagnostics (IVD) Expertise")
    st.markdown("My experience is grounded in a deep and practical understanding of the quality system regulations and standards essential to the IVD industry.")

    reg_option = st.selectbox(
        "Select a Regulation/Standard to see its practical application in my work:",
        ["21 CFR Part 820 (QSR)", "ISO 13485 (Medical Devices QMS)", "ISO 14971 (Risk Management)", "21 CFR Part 806 (Corrections & Removals)"]
    )

    if reg_option == "21 CFR Part 820 (QSR)":
        st.markdown("""
        **Focus:** The US FDA's Quality System Regulation for medical devices.
        **My Practical Application:**
        - **§820.100 (CAPA):** The entire 'Investigations' module is a demonstration of my CAPA methodology. I use statistical tools like SPC and ANOVA to "identify the cause" and Process Capability analysis to "verify... the corrective action... is effective," as mandated.
        - **§820.250 (Statistical Techniques):** I don't just use statistics; I validate their use. The Process Capability analysis (Cpk/Ppk) shown in the VOE section is a direct implementation of "procedures for identifying valid statistical techniques required for establishing, controlling, and verifying the acceptability of process capability."
        - **§820.30 (Design Controls):** My experience with DHF management (see portfolio) involves ensuring all design inputs, outputs, V&V activities, and risk analyses are meticulously documented and linked.
        """)
    elif reg_option == "ISO 13485:2016":
        st.markdown("""
        **Focus:** The international standard for a comprehensive QMS for medical devices.
        **My Practical Application:**
        - **7.3 (Design and Development):** My project management dashboards (e.g., Grifols V&V) are tools designed to manage the complexities of V&V planning, execution, and documentation as required.
        - **8.2.6 (Monitoring and Measurement of Product):** The SPC charts I use provide the "evidence of conformity with acceptance criteria" and ensure records identify the personnel authorizing release.
        - **8.4 (Analysis of Data):** The 'Continuous Improvement' module is a direct application of this clause. I use tools like Correlation Heatmaps and COPQ to analyze data from monitoring/measurement, feedback, and supplier performance to drive data-based decisions and demonstrate the suitability of the QMS.
        """)
    elif reg_option == "ISO 14971":
        st.markdown("""
        **Focus:** Application of risk management to medical devices throughout the product lifecycle.
        **My Practical Application:**
        - **Risk-Based CAPA:** My investigation process is inherently risk-based. The Pareto analysis helps prioritize CAPAs based on the highest frequency/risk defects. The scope of an investigation is always determined by a formal Product Impact Assessment.
        - **FMEA as an Input:** A Failure Modes and Effects Analysis (FMEA) from the design phase is a key input to my investigations. If a failure occurs, I first check if the failure mode was anticipated in the FMEA. If so, was the risk control effective? If not, the FMEA must be updated, ensuring a closed-loop risk management process.
        - **Risk Control Verification:** The VOE (Verification of Effectiveness) phase of a CAPA is also a verification of a risk control's effectiveness, a key requirement of ISO 14971.
        """)
    elif reg_option == "21 CFR Part 806":
        st.markdown("""
        **Focus:** Governs reporting requirements for actions taken to reduce a risk to health posed by a device.
        **My Practical Application:**
        - **The Link to CAPA:** A robust investigation is the foundation for a Part 806 decision. When a non-conformance is identified, a key part of the risk assessment is to determine if it could cause a "risk to health."
        - **Data-Driven Decisions:** My role is to provide the objective, statistical evidence to the cross-functional review board. For example, if a process shift results in out-of-specification product that could cause an incorrect patient result, I would present the control charts and capability data to quantify the scope and magnitude of the problem, enabling an informed and defensible decision on whether a reportable correction or removal is required.
        """)

def show_contact_module():
    st.header("Summary of Qualifications & Contact Information")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Education & Experience")
        st.markdown("""
        - **Education:** Ph.D./Master's in [Your Field - e.g., Bioengineering]
        - **Experience:** [X] years of progressive experience in Quality and Process Engineering within the Life Sciences and Diagnostics industries.
        - **Location:** Fully onsite in San Diego, CA.
        """)
        st.subheader("Key Competencies Demonstrated")
        st.markdown("""
        - **Quality Systems:** Data-Driven CAPA, NCEs, Complaints, Audits
        - **Advanced Statistical Analysis:** SPC, Cpk/Ppk, ANOVA, Pareto, Correlation
        - **Root Cause Analysis:** 5 Whys, Fishbone, FMEA, Hypothesis Testing
        - **Project Management:** Cross-functional team leadership, stakeholder comms
        - **Continuous Improvement:** Six Sigma (DMAIC), Lean, COPQ Analysis
        - **Regulatory Fluency:** Deep, practical knowledge of 21 CFR 820, ISO 13485, ISO 14971
        - **Technical Acumen:** IVD Product Lifecycle, Data Analysis (Python), Dashboarding (Streamlit)
        """)
    with col2:
        st.subheader("Let's Connect")
        st.markdown("""
        Thank you for taking the time to review my interactive application. I am confident that my skills and experience are an excellent match for the challenges and opportunities of the Quality Engineer role.
        
        I am eager to discuss how I can contribute to your team's success.
        """)
        st.markdown("---")
        st.markdown("### [Your Name]")
        st.markdown("**Email:** [your.email@email.com]")
        st.markdown("**LinkedIn:** [linkedin.com/in/yourprofile]")
        st.markdown("**Phone:** (555) 555-5555")

# --- Run the App ---
if __name__ == "__main__":
    run_app()
