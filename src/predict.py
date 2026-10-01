import os
import sys
import json
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import NUMERICAL_FEATURES, ModularColumnTransformerScaler
from train import LogisticRegressionClassifier, DecisionTreeClassifier, DecisionNode, RandomForestClassifier

REQUIRED_INPUT_KEYS = [
    'ssc_percentage', 'hsc_percentage', 'cgpa', 'coding_rating',
    'aptitude_score', 'soft_skills_score', 'internships',
    'projects_completed', 'workshops_attended', 'backlogs', 'gender'
]

VALID_INPUT_BOUNDS = {
    'ssc_percentage': (0.0, 100.0),
    'hsc_percentage': (0.0, 100.0),
    'cgpa': (0.0, 10.0),
    'coding_rating': (0, 1000),
    'aptitude_score': (0.0, 100.0),
    'soft_skills_score': (1.0, 5.0),
    'internships': (0, 10),
    'projects_completed': (0, 15),
    'workshops_attended': (0, 10),
    'backlogs': (0, 20)
}


def load_model_and_scaler():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')

    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model artifact missing at {best_model_path}. Run src/train.py first!")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler artifact missing at {scaler_path}. Run src/preprocessing.py first!")

    with open(best_model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def validate_student_input(student_dict):
    """
    Validates dictionary keys and range boundaries for user student input.
    """
    if not isinstance(student_dict, dict):
        raise TypeError("Student input must be a dictionary!")

    for key in REQUIRED_INPUT_KEYS:
        if key not in student_dict:
            raise KeyError(f"Missing required feature parameter: '{key}'")

    for key, (min_b, max_b) in VALID_INPUT_BOUNDS.items():
        val = float(student_dict[key])
        if val < min_b or val > max_b:
            raise ValueError(f"Value for '{key}' ({val}) is outside valid range [{min_b}, {max_b}]!")

    if student_dict['gender'] not in ['Male', 'Female', 'Other']:
        raise ValueError("Gender must be 'Male', 'Female', or 'Other'")

    return True


def predict_placement_outcome(student_dict):
    """
    Reusable prediction function. Takes a student profile dictionary, preprocesses features,
    and returns placement prediction class and model-estimated placement probability.
    """
    # 1. Validate Input
    validate_student_input(student_dict)

    # 2. Load Model & Scaler Artifacts (No retraining)
    model, scaler = load_model_and_scaler()

    # 3. Preprocess Input Vector using fitted Scaler
    scaled_vector = scaler.transform([student_dict])

    # 4. Model Probabilistic Inference
    proba = float(model.predict_proba(scaled_vector)[0])
    proba_pct = round(proba * 100, 2)
    
    prediction_label = "Placed" if proba >= 0.50 else "Not Placed"

    return {
        "status": "success",
        "predicted_class": prediction_label,
        "placement_probability_pct": proba_pct,
        "disclaimer": "This probability is a statistical estimate produced by the trained ML model based on historical training data. It does not guarantee employment or placement outcomes."
    }


if __name__ == '__main__':
    print("=" * 70)
    print("[TEST] TESTING REUSABLE PREDICTION FUNCTION (`src/predict.py`)")
    print("=" * 70)

    # Test Case 1: High Profile Student
    high_profile = {
        'ssc_percentage': 82.5,
        'hsc_percentage': 80.0,
        'cgpa': 8.4,
        'coding_rating': 750,
        'aptitude_score': 85.0,
        'soft_skills_score': 4.2,
        'internships': 2,
        'projects_completed': 3,
        'workshops_attended': 2,
        'backlogs': 0,
        'gender': 'Male'
    }

    print("\n[TEST CASE 1] High Profile Candidate:")
    res1 = predict_placement_outcome(high_profile)
    print(f"  - Predicted Class: {res1['predicted_class']}")
    print(f"  - Model Estimated Probability: {res1['placement_probability_pct']}%")

    # Test Case 2: At-Risk Candidate
    low_profile = {
        'ssc_percentage': 52.0,
        'hsc_percentage': 50.0,
        'cgpa': 5.1,
        'coding_rating': 210,
        'aptitude_score': 42.0,
        'soft_skills_score': 2.2,
        'internships': 0,
        'projects_completed': 0,
        'workshops_attended': 0,
        'backlogs': 2,
        'gender': 'Female'
    }

    print("\n[TEST CASE 2] At-Risk Candidate:")
    res2 = predict_placement_outcome(low_profile)
    print(f"  - Predicted Class: {res2['predicted_class']}")
    print(f"  - Model Estimated Probability: {res2['placement_probability_pct']}%")

    # Test Case 3: Validation Error Handling Test
    print("\n[TEST CASE 3] Out-of-Bounds Input Validation Error Test:")
    invalid_profile = high_profile.copy()
    invalid_profile['cgpa'] = 14.5  # Invalid CGPA > 10.0
    
    try:
        predict_placement_outcome(invalid_profile)
    except ValueError as err:
        print(f"  [+] Gracefully caught expected validation error: {err}")

    print("=" * 70)
