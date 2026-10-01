import os
import csv
import random

def generate_student_dataset(n_samples=1000, seed=42):
    random.seed(seed)
    
    headers = [
        'student_id', 'gender', 'ssc_percentage', 'hsc_percentage', 'cgpa',
        'coding_rating', 'aptitude_score', 'soft_skills_score',
        'internships', 'projects_completed', 'workshops_attended', 'backlogs', 'placement_status'
    ]
    
    rows = []
    
    for i in range(1, n_samples + 1):
        student_id = f"STU{1000 + i}"
        gender = 'Male' if random.random() < 0.58 else 'Female'
        
        # Performance distributions
        ssc_pct = round(min(max(random.gauss(72.5, 10.0), 45.0), 98.0), 2)
        hsc_pct = round(min(max(random.gauss(70.0, 11.0), 45.0), 98.0), 2)
        
        # CGPA correlated with school grades + noise
        base_cgpa = (ssc_pct + hsc_pct) / 20.0
        cgpa = round(min(max(base_cgpa + random.gauss(0.0, 0.8), 4.5), 10.0), 2)
        
        # Technical & Aptitude
        coding_rating = int(min(max(random.gauss(550 + (cgpa - 7.0) * 80, 120), 100), 1000))
        aptitude_score = round(min(max(random.gauss(65 + (cgpa - 7.0) * 5, 15), 30.0), 100.0), 1)
        soft_skills = round(min(max(random.gauss(3.5, 0.8), 1.0), 5.0), 1)
        
        internships = random.choices([0, 1, 2, 3, 4], weights=[0.35, 0.38, 0.18, 0.07, 0.02])[0]
        projects = random.choices([0, 1, 2, 3, 4, 5], weights=[0.10, 0.25, 0.35, 0.20, 0.07, 0.03])[0]
        workshops = random.choices([0, 1, 2, 3, 4], weights=[0.20, 0.40, 0.25, 0.10, 0.05])[0]
        backlogs = random.choices([0, 1, 2, 3, 4], weights=[0.72, 0.16, 0.08, 0.03, 0.01])[0]
        
        # Placement Logit Decision Formula based on industry criteria
        # High impact: CGPA >= 7.0, Coding >= 500, Aptitude >= 60, Internships >= 1, Backlogs == 0
        logit = (
            1.2 * (cgpa - 7.0) +
            0.004 * (coding_rating - 500) +
            0.03 * (aptitude_score - 60) +
            0.4 * (soft_skills - 3.0) +
            0.8 * internships +
            0.4 * projects +
            -1.1 * backlogs +
            0.015 * (ssc_pct - 60) +
            0.015 * (hsc_pct - 60) - 0.5
        )
        
        prob = 1.0 / (1.0 + random.expovariate(1.0) if logit < 0 else (1.0 + random.uniform(0.0, 0.3)))
        # Sigmoidal probability check
        import math
        sig_prob = 1.0 / (1.0 + math.exp(-logit))
        placed = 1 if sig_prob > 0.48 else 0
        placement_status = 'Placed' if placed == 1 else 'Not Placed'
        
        rows.append([
            student_id, gender, ssc_pct, hsc_pct, cgpa,
            coding_rating, aptitude_score, soft_skills,
            internships, projects, workshops, backlogs, placement_status
        ])
        
    return headers, rows

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    csv_path = os.path.join(raw_dir, 'student_data.csv')
    headers, rows = generate_student_dataset(1000)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"[+] Raw dataset generated at: {csv_path}")
    print(f"[+] Total Records: {len(rows)}")
