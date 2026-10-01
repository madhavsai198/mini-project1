import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import FEATURE_NAMES
from train import LogisticRegressionClassifier, DecisionTreeClassifier, DecisionNode, RandomForestClassifier


def load_inference_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')

    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    if not os.path.exists(best_model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or Scaler artifacts missing! Please run src/train.py first.")

    with open(best_model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_placement_probability(student_dict):
    """
    Predicts placement outcome and model-estimated placement probability for a student profile.
    
    Disclaimer: The returned probability is the machine learning model's statistical estimate
    based on training data patterns and does not guarantee employment.
    """
    model, scaler = load_inference_artifacts()

    # Extract raw feature vector in exact training feature order
    # ['ssc_percentage', 'hsc_percentage', 'cgpa', 'coding_rating', 'aptitude_score', 'soft_skills_score', 'internships', 'projects_completed', 'workshops_attended', 'backlogs', 'gender_encoded']
    raw_vector = [
        float(student_dict['ssc_percentage']),
        float(student_dict['hsc_percentage']),
        float(student_dict['cgpa']),
        float(student_dict['coding_rating']),
        float(student_dict['aptitude_score']),
        float(student_dict['soft_skills_score']),
        float(student_dict['internships']),
        float(student_dict['projects_completed']),
        float(student_dict['workshops_attended']),
        float(student_dict['backlogs']),
        1 if student_dict.get('gender', 'Male') == 'Male' else 0
    ]

    raw_array = np.array([raw_vector])
    scaled_array = scaler.transform(raw_array)

    proba = float(model.predict_proba(scaled_array)[0])
    proba_pct = round(proba * 100, 2)

    prediction_label = "Placed" if proba >= 0.50 else "Not Placed"

    return {
        "prediction": prediction_label,
        "estimated_probability_pct": proba_pct,
        "disclaimer": "This probability is a statistical estimate produced by the trained ML model based on historical training data. It does not guarantee employment or placement outcomes."
    }


if __name__ == '__main__':
    sample_student = {
        'ssc_percentage': 78.5,
        'hsc_percentage': 75.0,
        'cgpa': 7.8,
        'coding_rating': 620,
        'aptitude_score': 72.0,
        'soft_skills_score': 3.8,
        'internships': 1,
        'projects_completed': 2,
        'workshops_attended': 1,
        'backlogs': 0,
        'gender': 'Male'
    }

    result = predict_placement_probability(sample_student)
    print("[+] Test Sample Prediction:")
    print(f"    - Prediction: {result['prediction']}")
    print(f"    - Model Estimated Probability: {result['estimated_probability_pct']}%")
