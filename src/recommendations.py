import os

# Centralized Configurable Educational Thresholds (Easily customizable for institution criteria)
DEFAULT_THRESHOLDS = {
    "cgpa_min": 7.0,
    "cgpa_target": 8.0,
    "attendance_min": 75.0,
    "coding_score_min": 500,
    "coding_score_target": 650,
    "aptitude_score_min": 65.0,
    "communication_score_min": 3.5,
    "internships_min": 1,
    "projects_min": 2,
    "certifications_min": 1,
    "backlogs_max": 0
}


def generate_profile_recommendations(student_dict, thresholds=None):
    """
    Rule-based student profile improvement recommendation system.
    Identifies profile strengths and actionable improvement suggestions based on configurable thresholds.
    Completely decoupled from the ML prediction model logic.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    strengths = []
    improvements = []

    # Extract student values with safe numeric defaults
    cgpa = float(student_dict.get('cgpa', 0.0))
    attendance = float(student_dict.get('ssc_percentage', student_dict.get('hsc_percentage', 75.0)))  # Or attendance if provided
    coding = float(student_dict.get('coding_rating', student_dict.get('coding_score', 0)))
    aptitude = float(student_dict.get('aptitude_score', 0.0))
    communication = float(student_dict.get('soft_skills_score', student_dict.get('communication_score', 0.0)))
    internships = int(student_dict.get('internships', 0))
    projects = int(student_dict.get('projects_completed', student_dict.get('projects', 0)))
    certifications = int(student_dict.get('workshops_attended', student_dict.get('certifications', 0)))
    backlogs = int(student_dict.get('backlogs', 0))

    # 1. Academic CGPA Evaluation
    if cgpa >= thresholds["cgpa_target"]:
        strengths.append(f"Outstanding academic performance (CGPA: {cgpa:.2f} >= {thresholds['cgpa_target']:.1f}).")
    elif cgpa >= thresholds["cgpa_min"]:
        strengths.append(f"Good academic baseline meeting standard eligibility (CGPA: {cgpa:.2f} >= {thresholds['cgpa_min']:.1f}).")
    else:
        improvements.append({
            "category": "Academic Performance (CGPA)",
            "issue": f"Current CGPA is {cgpa:.2f}, below recommended {thresholds['cgpa_min']:.1f} eligibility cutoff.",
            "suggestion": "Focus on core subject coursework and assignments to raise CGPA above 7.0 to meet campus drive cutoffs."
        })

    # 2. Coding & Technical DSA Rating
    if coding >= thresholds["coding_score_target"]:
        strengths.append(f"Exceptional problem-solving & coding skill rating ({int(coding)} >= {thresholds['coding_score_target']}).")
    elif coding >= thresholds["coding_score_min"]:
        strengths.append(f"Solid coding foundation meeting placement technical standards ({int(coding)} >= {thresholds['coding_score_min']}).")
    else:
        improvements.append({
            "category": "Coding & Algorithm Practice",
            "issue": f"Coding rating of {int(coding)} is below competitive placement benchmark ({thresholds['coding_score_min']}).",
            "suggestion": "Practice Data Structures and Algorithms (DSA) daily on platforms like LeetCode, HackerRank, or CodeChef."
        })

    # 3. Quantitative & Logical Aptitude
    if aptitude >= thresholds["aptitude_score_min"]:
        strengths.append(f"Strong aptitude test preparedness ({aptitude:.1f}% >= {thresholds['aptitude_score_min']:.1f}%).")
    else:
        improvements.append({
            "category": "Aptitude & Reasoning",
            "issue": f"Aptitude score of {aptitude:.1f}% is below screening round benchmark ({thresholds['aptitude_score_min']:.1f}%).",
            "suggestion": "Solve daily quantitative aptitude, logical reasoning, and verbal practice modules to clear preliminary online screening tests."
        })

    # 4. Communication & Soft Skills
    if communication >= thresholds["communication_score_min"]:
        strengths.append(f"Good soft skills & communication rating ({communication:.1f} >= {thresholds['communication_score_min']:.1f}).")
    else:
        improvements.append({
            "category": "Communication & Interview Practice",
            "issue": f"Soft skills score is {communication:.1f} out of 5.0 (below {thresholds['communication_score_min']:.1f} target).",
            "suggestion": "Participate in mock HR interviews, group discussions (GD), and presentation workshops to build interview confidence."
        })

    # 5. Internships & Experience
    if internships >= thresholds["internships_min"]:
        strengths.append(f"Practical industry experience through {internships} completed internship(s).")
    else:
        improvements.append({
            "category": "Industry Experience",
            "issue": "Zero completed internships recorded in profile.",
            "suggestion": "Apply for summer internships, virtual micro-internships, or industrial training programs to gain hands-on industry exposure."
        })

    # 6. Projects & Portfolio
    if projects >= thresholds["projects_min"]:
        strengths.append(f"Demonstrated domain application with {projects} technical projects.")
    else:
        improvements.append({
            "category": "Project Portfolio",
            "issue": f"Only {projects} project(s) completed (recommended: {thresholds['projects_min']}+).",
            "suggestion": "Build 2 full-stack or ML capstone projects, document code cleanly on GitHub, and host live working demos."
        })

    # 7. Certifications & Skill Courses
    if certifications >= thresholds["certifications_min"]:
        strengths.append(f"Verified domain skill certifications completed ({certifications}).")
    else:
        improvements.append({
            "category": "Skill Certifications",
            "issue": "No additional domain certifications or workshops completed.",
            "suggestion": "Enroll in relevant specialization certifications (e.g. NPTEL, Coursera, AWS, or full-stack bootcamps)."
        })

    # 8. Academic Backlogs
    if backlogs <= thresholds["backlogs_max"]:
        strengths.append("Clean academic record with zero active backlogs.")
    else:
        improvements.append({
            "category": "Academic Backlogs",
            "issue": f"{backlogs} active/uncleared backlog(s) detected.",
            "suggestion": "Prioritize clearing active backlogs in upcoming supplementary exams, as most campus recruiters enforce a strict zero-backlog policy."
        })

    return {
        "strengths": strengths,
        "improvements": improvements
    }


if __name__ == '__main__':
    print("=" * 70)
    print("[TEST] TESTING RECOMMENDATION SYSTEM (`src/recommendations.py`)")
    print("=" * 70)

    sample_student = {
        'cgpa': 6.6,
        'coding_rating': 410,
        'aptitude_score': 55.0,
        'soft_skills_score': 3.0,
        'internships': 0,
        'projects_completed': 1,
        'workshops_attended': 0,
        'backlogs': 1
    }

    results = generate_profile_recommendations(sample_student)
    
    print("\n[+] Identified Candidate Strengths:")
    for s in results['strengths']:
        print(f"  - {s}")

    print("\n[+] Actionable Areas for Improvement:")
    for imp in results['improvements']:
        print(f"  - [{imp['category']}]")
        print(f"    Issue: {imp['issue']}")
        print(f"    Action: {imp['suggestion']}\n")

    print("=" * 70)
