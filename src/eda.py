import os
import csv
import json
import numpy as np


def compute_eda_summary(raw_csv_path):
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw CSV file not found: {raw_csv_path}")

    with open(raw_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [r for r in reader if r]

    total_students = len(rows)
    col_idx = {name: i for i, name in enumerate(headers)}

    placed_count = sum(1 for r in rows if r[col_idx['placement_status']] == 'Placed')
    not_placed_count = total_students - placed_count
    placement_rate = round((placed_count / total_students) * 100, 2)

    # Feature statistics calculation
    num_features = ['cgpa', 'coding_rating', 'aptitude_score', 'soft_skills_score', 'internships', 'projects_completed', 'backlogs']
    
    stats = {}
    for feat in num_features:
        placed_vals = [float(r[col_idx[feat]]) for r in rows if r[col_idx['placement_status']] == 'Placed']
        not_placed_vals = [float(r[col_idx[feat]]) for r in rows if r[col_idx['placement_status']] == 'Not Placed']
        all_vals = placed_vals + not_placed_vals
        
        stats[feat] = {
            "overall_mean": round(float(np.mean(all_vals)), 2),
            "placed_mean": round(float(np.mean(placed_vals)), 2),
            "not_placed_mean": round(float(np.mean(not_placed_vals)), 2),
            "min": round(float(np.min(all_vals)), 2),
            "max": round(float(np.max(all_vals)), 2)
        }

    summary = {
        "total_students": total_students,
        "placed_count": placed_count,
        "not_placed_count": not_placed_count,
        "placement_rate_pct": placement_rate,
        "feature_statistics": stats
    }
    return summary


def run_eda_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, 'data', 'raw', 'student_data.csv')
    output_json = os.path.join(base_dir, 'data', 'processed', 'eda_summary.json')

    summary = compute_eda_summary(raw_csv)
    
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=4)

    print("[+] EDA Summary generated successfully:")
    print(f"    - Total Students: {summary['total_students']}")
    print(f"    - Placed Count: {summary['placed_count']} ({summary['placement_rate_pct']}%)")
    print(f"    - Placed vs Not Placed CGPA: {summary['feature_statistics']['cgpa']['placed_mean']} vs {summary['feature_statistics']['cgpa']['not_placed_mean']}")
    print(f"[+] Saved summary report to: {output_json}")
    return summary


if __name__ == '__main__':
    run_eda_pipeline()
