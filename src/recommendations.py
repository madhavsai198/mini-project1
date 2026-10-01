def generate_profile_recommendations(student_dict):
    """
    Rule-based profile analysis and actionable improvement suggestions.
    Completely decoupled from the ML prediction model logic.
    """
    strengths = []
    improvements = []

    cgpa = float(student_dict.get('cgpa', 0))
    coding = float(student_dict.get('coding_rating', 0))
    aptitude = float(student_dict.get('aptitude_score', 0))
    soft_skills = float(student_dict.get('soft_skills_score', 0))
    internships = int(student_dict.get('internships', 0))
    projects = int(student_dict.get('projects_completed', 0))
    backlogs = int(student_dict.get('backlogs', 0))

    # 1. Academic Performance (CGPA)
    if cgpa >= 8.0:
        strengths.append("Excellent academic consistency (CGPA >= 8.0).")
    elif cgpa >= 7.0:
        strengths.append("Good academic baseline (CGPA >= 7.0).")
    else:
        improvements.append({
            "category": "Academic Performance",
            "issue": f"Current CGPA is {cgpa:.2f}, below the recommended 7.0 campus eligibility cutoff.",
            "suggestion": "Focus on upcoming semester coursework and assignments to raise CGPA above 7.0 to meet company eligibility criteria."
        })

    # 2. Coding & Technical Skills
    if coding >= 650:
        strengths.append("Strong problem-solving & coding skills (Rating >= 650).")
    elif coding >= 500:
        strengths.append("Moderate coding foundation (Rating >= 500).")
    else:
        improvements.append({
            "category": "Coding & Algorithms",
            "issue": f"Coding rating of {int(coding)} is below competitive placement standards (500+).",
            "suggestion": "Practice Data Structures and Algorithms (DSA) daily on platforms like LeetCode, HackerRank, or CodeChef."
        })

    # 3. Aptitude & Reasoning
    if aptitude >= 75.0:
        strengths.append("High aptitude test preparedness (Score >= 75%).")
    elif aptitude >= 60.0:
        strengths.append("Decent quantitative aptitude foundation.")
    else:
        improvements.append({
            "category": "Quantitative & Logical Aptitude",
            "issue": f"Aptitude score of {aptitude:.1f}% is low for initial screening rounds.",
            "suggestion": "Solve daily quantitative aptitude, logical reasoning, and verbal practice sets to clear preliminary online screening tests."
        })

    # 4. Soft Skills & Communication
    if soft_skills >= 4.0:
        strengths.append("Outstanding communication & soft skills rating.")
    elif soft_skills < 3.5:
        improvements.append({
            "category": "Soft Skills & Communication",
            "issue": f"Soft skills rating is {soft_skills:.1f} out of 5.0.",
            "suggestion": "Participate in mock HR interviews, group discussions (GD), and presentation workshops to boost confidence."
        })

    # 5. Internships & Practical Experience
    if internships >= 1:
        strengths.append(f"Practical industry experience through {internships} internship(s).")
    else:
        improvements.append({
            "category": "Industry Experience",
            "issue": "Zero completed internships recorded.",
            "suggestion": "Apply for summer internships, virtual micro-internships, or research assistantships to gain hands-on exposure."
        })

    # 6. Projects & Portfolio
    if projects >= 2:
        strengths.append(f"Demonstrated domain application with {projects} technical projects.")
    else:
        improvements.append({
            "category": "Project Portfolio",
            "issue": "Fewer than 2 technical projects completed.",
            "suggestion": "Build 2 full-stack or ML capstone projects, document them cleanly on GitHub, and deploy live demos."
        })

    # 7. Academic Backlogs
    if backlogs == 0:
        strengths.append("Clean academic record with zero active backlogs.")
    else:
        improvements.append({
            "category": "Academic Backlogs",
            "issue": f"{backlogs} active/uncleared backlog(s) detected.",
            "suggestion": "Prioritize clearing all active backlogs in the immediate supplementary exams, as most recruiters mandate zero active backlogs."
        })

    return {
        "strengths": strengths,
        "improvements": improvements
    }


if __name__ == '__main__':
    sample = {
        'cgpa': 6.5,
        'coding_rating': 420,
        'aptitude_score': 55.0,
        'soft_skills_score': 3.0,
        'internships': 0,
        'projects_completed': 1,
        'backlogs': 1
    }
    recs = generate_profile_recommendations(sample)
    print(f"[+] Identified Strengths: {len(recs['strengths'])}")
    print(f"[+] Profile Areas for Improvement: {len(recs['improvements'])}")
