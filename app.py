import os
import sys
import json
import streamlit as st
import numpy as np

# Ensure src directory is in Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from predict import predict_placement_outcome
from recommendations import generate_profile_recommendations
from eda import compute_eda_summary

# Streamlit Page Configuration
st.set_page_config(
    page_title="Student Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
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
    st.markdown('<div class="main-header">🎓 Student Performance & Placement Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Supervised Machine Learning Diagnostic Dashboard & Career Readiness System</div>', unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("📌 Navigation")
    menu = st.sidebar.radio(
        "Select Feature Section:",
        ["🎯 Placement Predictor", "📊 Exploratory Data Analysis (EDA)", "🤖 ML Model Comparison", "💡 Profile Improvement Suggestions"]
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv = os.path.join(base_dir, 'data', 'raw', 'student_data.csv')
    eval_json = os.path.join(base_dir, 'models', 'evaluation_results.json')

    # Load summary and evaluation data
    eda_summary = compute_eda_summary(raw_csv) if os.path.exists(raw_csv) else None
    
    eval_data = None
    if os.path.exists(eval_json):
        with open(eval_json, 'r') as f:
            eval_data = json.load(f)

    # SECTION 1: PLACEMENT PREDICTOR
    if menu == "🎯 Placement Predictor":
        st.header("🎯 Student Placement Risk & Probability Predictor")
        st.write("Enter the student's academic performance, technical ratings, and extracurricular attributes below.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📚 Academic & Qualification Metrics")
            cgpa = st.slider("College CGPA (scale 4.0 - 10.0)", 4.0, 10.0, 7.5, step=0.1)
            ssc_pct = st.slider("10th (SSC) Marks Percentage", 45.0, 98.0, 75.0, step=0.5)
            hsc_pct = st.slider("12th (HSC) Marks Percentage", 45.0, 98.0, 72.0, step=0.5)
            backlogs = st.number_input("Active/Uncleared Backlogs", min_value=0, max_value=6, value=0)
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col2:
            st.subheader("💻 Skill Ratings & Experience")
            coding_rating = st.slider("Coding / DSA Skill Rating (100 - 1000)", 100, 1000, 600, step=10)
            aptitude_score = st.slider("Quantitative Aptitude Score (%)", 30.0, 100.0, 70.0, step=1.0)
            soft_skills = st.slider("Soft Skills & Communication (1.0 - 5.0)", 1.0, 5.0, 3.8, step=0.1)
            internships = st.number_input("Internships Completed", min_value=0, max_value=5, value=1)
            projects = st.number_input("Technical Projects Completed", min_value=0, max_value=6, value=2)
            workshops = st.number_input("Workshops / Certifications Attended", min_value=0, max_value=5, value=1)

        student_payload = {
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

        st.markdown("---")
        if st.button("🚀 Predict Placement Outcome", type="primary", use_container_width=True):
            try:
                res = predict_placement_outcome(student_payload)
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if res['predicted_class'] == "Placed":
                        st.success(f"### Predicted Outcome: **{res['predicted_class']}** 🟢")
                    else:
                        st.error(f"### Predicted Outcome: **{res['predicted_class']}** 🔴")
                
                with res_col2:
                    st.metric(
                        label="Model Estimated Placement Probability",
                        value=f"{res['placement_probability_pct']}%"
                    )

                st.progress(res['placement_probability_pct'] / 100.0)

                # Mandatory Disclaimer Notice
                st.markdown(
                    f'<div class="disclaimer-box"><strong>⚠️ Important Disclaimer:</strong> {res["disclaimer"]}</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    # SECTION 2: EDA
    elif menu == "📊 Exploratory Data Analysis (EDA)":
        st.header("📊 Student Dataset Exploratory Analysis")
        
        if eda_summary:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Students Analyzed", eda_summary['total_students'])
            m2.metric("Placed Count", eda_summary['placed_count'])
            m3.metric("Not Placed Count", eda_summary['not_placed_count'])
            m4.metric("Placement Success Rate", f"{eda_summary['placement_rate_pct']}%")

            st.markdown("---")
            st.subheader("📌 Key Attribute Comparison: Placed vs. Non-Placed Students")
            
            stats = eda_summary['feature_statistics']
            comp_data = []
            for feat, val in stats.items():
                comp_data.append({
                    "Feature Attribute": feat.replace('_', ' ').title(),
                    "Placed Students Mean": val['placed_mean'],
                    "Not Placed Students Mean": val['not_placed_mean'],
                    "Overall Average": val['overall_mean']
                })
            
            st.dataframe(comp_data, use_container_width=True)

    # SECTION 3: MODEL COMPARISON
    elif menu == "🤖 ML Model Comparison":
        st.header("🤖 Supervised Classification Model Benchmarks")
        st.write("Comparison of 3 machine learning algorithms trained and evaluated on 200 unseen validation students.")

        if eval_data:
            st.info(f"🏆 **Dynamically Selected Best Model**: **{eval_data['best_model'].upper()}** (Evaluated by test set F1-Score)")

            models_dict = eval_data['models']
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

    # SECTION 4: RECOMMENDATIONS
    elif menu == "💡 Profile Improvement Suggestions":
        st.header("💡 Student Profile Improvement & Action Plan")
        st.write("Rule-based analysis identifying profile strengths and areas for development.")

        st.subheader("Input Profile Values")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            r_cgpa = st.number_input("CGPA", 4.0, 10.0, 6.8, step=0.1, key="r_cgpa")
            r_coding = st.number_input("Coding Rating", 100, 1000, 420, step=10, key="r_coding")
            r_apt = st.number_input("Aptitude Score (%)", 30.0, 100.0, 58.0, step=1.0, key="r_apt")
            r_soft = st.number_input("Soft Skills Score (1-5)", 1.0, 5.0, 3.2, step=0.1, key="r_soft")
        with r_col2:
            r_intern = st.number_input("Internships", 0, 5, 0, key="r_intern")
            r_proj = st.number_input("Projects", 0, 6, 1, key="r_proj")
            r_back = st.number_input("Active Backlogs", 0, 6, 1, key="r_back")

        r_payload = {
            'cgpa': r_cgpa,
            'coding_rating': r_coding,
            'aptitude_score': r_apt,
            'soft_skills_score': r_soft,
            'internships': r_intern,
            'projects_completed': r_proj,
            'backlogs': r_back
        }

        if st.button("🔍 Analyze Profile & Generate Recommendations", use_container_width=True):
            recs = generate_profile_recommendations(r_payload)
            
            st.markdown("### ✅ Strengths Identified")
            if recs['strengths']:
                for s in recs['strengths']:
                    st.success(f"• {s}")
            else:
                st.write("No major strengths recorded yet.")

            st.markdown("### ⚠️ Actionable Areas for Improvement")
            if recs['improvements']:
                for imp in recs['improvements']:
                    with st.expander(f"📌 {imp['category']}: {imp['issue']}", expanded=True):
                        st.write(f"**Action Plan**: {imp['suggestion']}")
            else:
                st.info("Great profile! All key eligibility and skill parameters meet competitive standards.")


if __name__ == '__main__':
    main()
