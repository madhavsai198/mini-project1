import os
import sys
import json
import streamlit as st
import numpy as np

# Ensure src directory is in Python path for clean module imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from predict import predict_placement_outcome
from recommendations import generate_profile_recommendations
from eda import compute_eda_summary

# Streamlit Page Setup
st.set_page_config(
    page_title="Student Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #F59E0B;
        color: #92400E;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-title">🎓 Student Performance & Placement Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">College Mini Project — Supervised Machine Learning & Career Analytics Dashboard</div>', unsafe_allow_html=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv = os.path.join(base_dir, 'data', 'raw', 'student_data.csv')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    eval_json = os.path.join(base_dir, 'models', 'evaluation_results.json')
    config_json = os.path.join(base_dir, 'models', 'model_config.json')

    # Load EDA & Evaluation data
    eda_summary = compute_eda_summary(raw_csv) if os.path.exists(raw_csv) else None
    
    eval_data = None
    if os.path.exists(eval_json):
        with open(eval_json, 'r') as f:
            eval_data = json.load(f)

    model_config = None
    if os.path.exists(config_json):
        with open(config_json, 'r') as f:
            model_config = json.load(f)

    # Sidebar Navigation
    st.sidebar.title("📌 Dashboard Navigation")
    menu = st.sidebar.radio(
        "Select Section:",
        [
            "1. 📌 Overview",
            "2. 📊 Data Analysis",
            "3. 🤖 Model Performance",
            "4. 🎯 Placement Prediction",
            "5. 💡 Improvement Areas"
        ]
    )

    # SECTION 1: OVERVIEW
    if menu == "1. 📌 Overview":
        st.header("📌 Project Overview & Key Student Metrics")
        st.write("Summary statistics extracted from the student placement dataset.")

        if eda_summary:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Students Analyzed", eda_summary['total_students'])
            m2.metric("Number Placed (Class 1)", eda_summary['placed_count'])
            m3.metric("Number Not Placed (Class 0)", eda_summary['not_placed_count'])

            st.markdown("---")
            m4, m5, m6 = st.columns(3)
            m4.metric("Placement Success Rate", f"{eda_summary['placement_rate_pct']}%")
            
            stats = eda_summary['feature_statistics']
            avg_cgpa = stats.get('cgpa', {}).get('overall_mean', 7.12)
            avg_coding = stats.get('coding_rating', {}).get('overall_mean', 558.4)
            
            m5.metric("Average College CGPA", f"{avg_cgpa:.2f} / 10.0")
            m6.metric("Average Coding Rating", f"{avg_coding:.1f} / 1000")

            st.markdown("---")
            st.subheader("📋 Dataset Feature Attributes")
            st.write("The system evaluates **11 core features** covering academic, technical, aptitude, soft skills, and experience factors:")
            
            feat_cols = st.columns(2)
            with feat_cols[0]:
                st.markdown("""
                - **College CGPA**: Cumulative Grade Point Average (4.0 - 10.0)
                - **10th (SSC) Marks %**: Secondary School Percentage (40% - 100%)
                - **12th (HSC) Marks %**: Higher Secondary Percentage (40% - 100%)
                - **Coding Rating**: Technical DSA Problem Solving Score (100 - 1000)
                - **Aptitude Score**: Quantitative & Logical Reasoning Score (30% - 100%)
                - **Soft Skills Rating**: Communication & HR Interview Rating (1.0 - 5.0)
                """)
            with feat_cols[1]:
                st.markdown("""
                - **Internships Completed**: Industrial internship count (0 - 5)
                - **Projects Completed**: Technical capstone project count (0 - 6)
                - **Workshops Attended**: Certifications & technical workshops (0 - 5)
                - **Active Backlogs**: Uncleared failed subjects (0 - 6)
                - **Gender**: Student gender classification (`Male` / `Female`)
                """)

    # SECTION 2: DATA ANALYSIS
    elif menu == "2. 📊 Data Analysis":
        st.header("📊 Exploratory Data Analysis & Visualizations")
        st.write("Visual breakdown of student factors influencing placement outcomes.")

        tab1, tab2, tab3 = st.tabs(["🎯 Placement & Demographics", "📈 Academic & Skill Factors", "🔥 Correlation Heatmap"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Target Placement Distribution")
                img_path = os.path.join(figures_dir, 'placement_distribution.png')
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
                else:
                    st.info("Run `python src/eda.py` to generate visualization plots.")
            with c2:
                st.subheader("10th Marks Distribution vs Placement")
                img_path = os.path.join(figures_dir, 'attendance_vs_placement.png')
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("CGPA vs. Placement Status")
                img_path = os.path.join(figures_dir, 'cgpa_vs_placement.png')
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)

                st.subheader("Internships Completed vs Placement")
                img_path = os.path.join(figures_dir, 'internships_vs_placement.png')
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)

            with col2:
                st.subheader("Coding DSA Rating vs Placement")
                img_path = os.path.join(figures_dir, 'coding_score_vs_placement.png')
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)

                st.subheader("Active Backlogs vs Placement")
                img_path = os.path.join(figures_dir, 'backlogs_vs_placement.png')
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)

        with tab3:
            st.subheader("Pairwise Feature Correlation Matrix")
            img_path = os.path.join(figures_dir, 'correlation_heatmap.png')
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)

    # SECTION 3: MODEL PERFORMANCE
    elif menu == "3. 🤖 Model Performance":
        st.header("🤖 Machine Learning Model Evaluation & Comparison")
        st.write("Supervised classification models evaluated on 199 unseen validation test students.")

        if model_config:
            best_m = model_config.get('best_model_name', 'logistic_regression').replace('_', ' ').title()
            st.success(f"🏆 **Selected Best Model**: **{best_m}** (Dynamically selected based on test set F1-Score)")

        if eval_data:
            st.subheader("📊 Model Comparison Table")
            models_dict = eval_data.get('models', {})
            table_rows = []
            
            for m_name, m_metrics in models_dict.items():
                table_rows.append({
                    "Model Algorithm": m_name.replace('_', ' ').title(),
                    "Accuracy (%)": f"{m_metrics['accuracy'] * 100:.2f}%",
                    "Precision": m_metrics['precision'],
                    "Recall": m_metrics['recall'],
                    "F1-Score": m_metrics['f1_score'],
                    "ROC-AUC": m_metrics['roc_auc']
                })

            st.table(table_rows)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("Confusion Matrices (Test Set)")
            img_path = os.path.join(figures_dir, 'confusion_matrices.png')
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)

        with col_m2:
            st.subheader("ROC Curve Model Comparison")
            img_path = os.path.join(figures_dir, 'roc_curve_comparison.png')
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)

    # SECTION 4: PLACEMENT PREDICTION
    elif menu == "4. 🎯 Placement Prediction":
        st.header("🎯 Student Placement Prediction Form")
        st.write("Enter student candidate metrics to compute predicted placement status and estimated probability.")

        with st.form("prediction_input_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📚 Academic Record")
                cgpa = st.number_input("College CGPA (4.0 - 10.0)", 4.0, 10.0, 7.6, step=0.1)
                ssc_pct = st.number_input("10th (SSC) Marks %", 40.0, 100.0, 76.5, step=0.5)
                hsc_pct = st.number_input("12th (HSC) Marks %", 40.0, 100.0, 74.0, step=0.5)
                backlogs = st.number_input("Active Backlogs Count", 0, 10, 0)
                gender = st.selectbox("Gender", ["Male", "Female"])

            with col2:
                st.subheader("💻 Skill Ratings & Experience")
                coding_rating = st.number_input("Coding / DSA Skill Rating (100 - 1000)", 100, 1000, 620, step=10)
                aptitude_score = st.number_input("Aptitude Test Score (%)", 30.0, 100.0, 72.0, step=1.0)
                soft_skills = st.number_input("Soft Skills & HR Rating (1.0 - 5.0)", 1.0, 5.0, 3.8, step=0.1)
                internships = st.number_input("Internships Completed", 0, 5, 1)
                projects = st.number_input("Projects Completed", 0, 8, 2)
                workshops = st.number_input("Workshops / Certifications", 0, 5, 1)

            submit_btn = st.form_submit_button("🚀 Run Placement Prediction", use_container_width=True)

        if submit_btn:
            payload = {
                'ssc_percentage': ssc_pct,
                'hsc_percentage': hsc_pct,
                'cgpa': cgpa,
                'coding_rating': coding_rating,
                'aptitude_score': aptitude_score,
                'soft_skills_score': soft_skills,
                'internships': internships,
                'projects_completed': projects,
                'workshops_attended': workshops,
                'backlogs': backlogs,
                'gender': gender
            }

            try:
                res = predict_placement_outcome(payload)

                st.markdown("---")
                st.subheader("📋 Prediction Results Summary")

                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    if res['predicted_class'] == "Placed":
                        st.success(f"### Predicted Status: **{res['predicted_class']}** 🟢")
                    else:
                        st.error(f"### Predicted Status: **{res['predicted_class']}** 🔴")

                with r_col2:
                    st.metric(
                        label="Model Estimated Placement Probability",
                        value=f"{res['placement_probability_pct']}%"
                    )

                st.progress(res['placement_probability_pct'] / 100.0)

                with st.expander("👤 Submitted Student Profile Summary", expanded=True):
                    st.json(payload)

                # Mandatory Ethical Disclaimer
                st.markdown(
                    f'<div class="disclaimer-box"><strong>⚠️ Disclaimer:</strong> {res["disclaimer"]}</div>',
                    unsafe_allow_html=True
                )

            except Exception as err:
                st.error(f"Prediction Input Error: {str(err)}")

    # SECTION 5: IMPROVEMENT AREAS
    elif menu == "5. 💡 Improvement Areas":
        st.header("💡 Profile Weakness Identification & Action Plan")
        st.write("Rule-based profile analysis identifying candidate strengths and actionable improvement suggestions.")

        with st.form("recommendation_form"):
            r_c1, r_c2 = st.columns(2)
            with r_c1:
                r_cgpa = st.number_input("CGPA", 4.0, 10.0, 6.6, step=0.1)
                r_coding = st.number_input("Coding Rating", 100, 1000, 420, step=10)
                r_apt = st.number_input("Aptitude Score (%)", 30.0, 100.0, 55.0, step=1.0)
                r_soft = st.number_input("Soft Skills Score (1-5)", 1.0, 5.0, 3.0, step=0.1)
            with r_c2:
                r_intern = st.number_input("Internships Completed", 0, 5, 0)
                r_proj = st.number_input("Projects Completed", 0, 8, 1)
                r_work = st.number_input("Certifications", 0, 5, 0)
                r_back = st.number_input("Active Backlogs", 0, 10, 1)

            rec_btn = st.form_submit_button("🔍 Analyze Profile & Generate Recommendations", use_container_width=True)

        if rec_btn:
            r_payload = {
                'cgpa': r_cgpa,
                'coding_rating': r_coding,
                'aptitude_score': r_apt,
                'soft_skills_score': r_soft,
                'internships': r_intern,
                'projects_completed': r_proj,
                'workshops_attended': r_work,
                'backlogs': r_back
            }

            recs = generate_profile_recommendations(r_payload)

            st.markdown("### ✅ Identified Candidate Strengths")
            if recs['strengths']:
                for s in recs['strengths']:
                    st.success(f"• {s}")
            else:
                st.info("No major strengths recorded yet.")

            st.markdown("### ⚠️ Actionable Areas for Improvement")
            if recs['improvements']:
                for imp in recs['improvements']:
                    with st.expander(f"📌 {imp['category']}: {imp['issue']}", expanded=True):
                        st.write(f"**Actionable Advice**: {imp['suggestion']}")
            else:
                st.success("Great profile! All key eligibility and technical parameters meet competitive placement standards.")


if __name__ == '__main__':
    main()
