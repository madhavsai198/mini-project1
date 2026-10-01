import os
import csv
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless script execution
import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic theme
plt.style.use('ggplot')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 10,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8
})

NUMERICAL_FEATURES = [
    'ssc_percentage', 'hsc_percentage', 'cgpa', 'coding_rating',
    'aptitude_score', 'soft_skills_score', 'internships',
    'projects_completed', 'workshops_attended', 'backlogs'
]

FEATURE_LABELS = {
    'ssc_percentage': '10th (SSC) Marks %',
    'hsc_percentage': '12th (HSC) Marks %',
    'cgpa': 'College CGPA',
    'coding_rating': 'Coding / DSA Rating',
    'aptitude_score': 'Aptitude Test Score (%)',
    'soft_skills_score': 'Soft Skills Rating (1-5)',
    'internships': 'Internships Completed',
    'projects_completed': 'Projects Completed',
    'workshops_attended': 'Workshops Attended',
    'backlogs': 'Active Backlogs'
}


def load_cleaned_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned dataset missing: {csv_path}. Run src/preprocessing.py first!")

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [r for r in reader if r]

    col_idx = {name: i for i, name in enumerate(headers)}
    return headers, data, col_idx


def compute_descriptive_stats(data, col_idx):
    stats = {}
    for feat in NUMERICAL_FEATURES:
        vals = [float(r[col_idx[feat]]) for r in data]
        stats[feat] = {
            "mean": round(float(np.mean(vals)), 2),
            "std": round(float(np.std(vals)), 2),
            "median": round(float(np.median(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2),
            "q25": round(float(np.percentile(vals, 25)), 2),
            "q75": round(float(np.percentile(vals, 75)), 2)
        }
    return stats


def compute_eda_summary(raw_csv_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_data.csv')
    if not os.path.exists(cleaned_path):
        cleaned_path = raw_csv_path
        
    headers, data, col_idx = load_cleaned_dataset(cleaned_path)
    
    total_students = len(data)
    placed_count = sum(1 for r in data if int(r[col_idx['placement_status']]) == 1 or r[col_idx['placement_status']] == 'Placed')
    not_placed_count = total_students - placed_count
    placement_rate = round((placed_count / total_students) * 100, 2) if total_students > 0 else 0.0

    stats = {}
    for feat in NUMERICAL_FEATURES:
        placed_vals = [float(r[col_idx[feat]]) for r in data if int(r[col_idx['placement_status']]) == 1 or r[col_idx['placement_status']] == 'Placed']
        not_placed_vals = [float(r[col_idx[feat]]) for r in data if int(r[col_idx['placement_status']]) == 0 or r[col_idx['placement_status']] == 'Not Placed']
        all_vals = placed_vals + not_placed_vals
        
        stats[feat] = {
            "overall_mean": round(float(np.mean(all_vals)), 2) if all_vals else 0.0,
            "placed_mean": round(float(np.mean(placed_vals)), 2) if placed_vals else 0.0,
            "not_placed_mean": round(float(np.mean(not_placed_vals)), 2) if not_placed_vals else 0.0,
            "min": round(float(np.min(all_vals)), 2) if all_vals else 0.0,
            "max": round(float(np.max(all_vals)), 2) if all_vals else 0.0
        }

    return {
        "total_students": total_students,
        "placed_count": placed_count,
        "not_placed_count": not_placed_count,
        "placement_rate_pct": placement_rate,
        "feature_statistics": stats
    }


def compute_correlation_matrix(data, col_idx):
    feats = NUMERICAL_FEATURES + ['placement_status']
    matrix = []
    for r in data:
        row_vals = [float(r[col_idx[f]]) for f in feats]
        matrix.append(row_vals)
    
    np_matrix = np.array(matrix)
    corr = np.corrcoef(np_matrix, rowvar=False)
    return feats, corr


def generate_visualizations(data, col_idx, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    saved_plots = []

    # Extract placement mask
    placed_mask = np.array([int(r[col_idx['placement_status']]) for r in data]) == 1
    
    # 1. Placement Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = [np.sum(~placed_mask), np.sum(placed_mask)]
    bars = ax.bar(['Not Placed (0)', 'Placed (1)'], counts, color=['#e11d48', '#10b981'], width=0.45)
    ax.set_title('Target Distribution: Placement Status', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Students')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height} ({height/len(data)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(figures_dir, 'placement_distribution.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    saved_plots.append(('Placement Target Distribution', p1))

    # 2. CGPA vs Placement Status (Boxplot)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cgpa_placed = [float(r[col_idx['cgpa']]) for r in data if int(r[col_idx['placement_status']]) == 1]
    cgpa_not_placed = [float(r[col_idx['cgpa']]) for r in data if int(r[col_idx['placement_status']]) == 0]
    
    bp = ax.boxplot([cgpa_not_placed, cgpa_placed], patch_artist=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Not Placed (0)', 'Placed (1)'])
    bp['boxes'][0].set_facecolor('#fecdd3')
    bp['boxes'][1].set_facecolor('#a7f3d0')
    ax.set_title('College CGPA vs. Placement Status', fontsize=12, fontweight='bold')
    ax.set_ylabel('College CGPA (Scale 4.0 - 10.0)')
    plt.tight_layout()
    p2 = os.path.join(figures_dir, 'cgpa_vs_placement.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    saved_plots.append(('CGPA vs Placement', p2))

    # 3. 10th Marks vs Placement Status
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ssc_placed = [float(r[col_idx['ssc_percentage']]) for r in data if int(r[col_idx['placement_status']]) == 1]
    ssc_not_placed = [float(r[col_idx['ssc_percentage']]) for r in data if int(r[col_idx['placement_status']]) == 0]
    
    ax.hist(ssc_placed, bins=15, alpha=0.6, label='Placed', color='#10b981', density=True)
    ax.hist(ssc_not_placed, bins=15, alpha=0.6, label='Not Placed', color='#e11d48', density=True)
    ax.set_title('10th (SSC) Marks % Distribution by Placement Status', fontsize=12, fontweight='bold')
    ax.set_xlabel('10th Marks Percentage (%)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    p3 = os.path.join(figures_dir, 'attendance_vs_placement.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    saved_plots.append(('10th Marks vs Placement', p3))

    # 4. Coding Score vs Placement (Boxplot)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    coding_placed = [float(r[col_idx['coding_rating']]) for r in data if int(r[col_idx['placement_status']]) == 1]
    coding_not_placed = [float(r[col_idx['coding_rating']]) for r in data if int(r[col_idx['placement_status']]) == 0]
    
    bp = ax.boxplot([coding_not_placed, coding_placed], patch_artist=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Not Placed (0)', 'Placed (1)'])
    bp['boxes'][0].set_facecolor('#fecdd3')
    bp['boxes'][1].set_facecolor('#a7f3d0')
    ax.set_title('Coding / DSA Skill Rating vs. Placement Status', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coding Rating (100 - 1000)')
    plt.tight_layout()
    p4 = os.path.join(figures_dir, 'coding_score_vs_placement.png')
    plt.savefig(p4, dpi=300)
    plt.close()
    saved_plots.append(('Coding Rating vs Placement', p4))

    # 5. Internships vs Placement (Bar Plot)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    intern_placed = [int(r[col_idx['internships']]) for r in data if int(r[col_idx['placement_status']]) == 1]
    intern_not_placed = [int(r[col_idx['internships']]) for r in data if int(r[col_idx['placement_status']]) == 0]
    
    counts_placed = [intern_placed.count(i) for i in range(5)]
    counts_not_placed = [intern_not_placed.count(i) for i in range(5)]
    
    x = np.arange(5)
    width = 0.35
    ax.bar(x - width/2, counts_not_placed, width, label='Not Placed', color='#e11d48')
    ax.bar(x + width/2, counts_placed, width, label='Placed', color='#10b981')
    ax.set_title('Internships Completed vs. Placement Outcome', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Internships Completed')
    ax.set_ylabel('Student Count')
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    p5 = os.path.join(figures_dir, 'internships_vs_placement.png')
    plt.savefig(p5, dpi=300)
    plt.close()
    saved_plots.append(('Internships vs Placement', p5))

    # 6. Backlogs vs Placement (Bar Plot)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    back_placed = [int(r[col_idx['backlogs']]) for r in data if int(r[col_idx['placement_status']]) == 1]
    back_not_placed = [int(r[col_idx['backlogs']]) for r in data if int(r[col_idx['placement_status']]) == 0]
    
    b_counts_placed = [back_placed.count(i) for i in range(5)]
    b_counts_not_placed = [back_not_placed.count(i) for i in range(5)]
    
    ax.bar(x - width/2, b_counts_not_placed, width, label='Not Placed', color='#e11d48')
    ax.bar(x + width/2, b_counts_placed, width, label='Placed', color='#10b981')
    ax.set_title('Active Backlogs vs. Placement Outcome', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Active Backlogs')
    ax.set_ylabel('Student Count')
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    p6 = os.path.join(figures_dir, 'backlogs_vs_placement.png')
    plt.savefig(p6, dpi=300)
    plt.close()
    saved_plots.append(('Backlogs vs Placement', p6))

    # 7. Correlation Heatmap using Matplotlib imshow
    feats, corr = compute_correlation_matrix(data, col_idx)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)
    
    short_labels = [f.replace('_', ' ').title() for f in feats]
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(short_labels, rotation=45, ha='left', fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    
    for i in range(len(feats)):
        for j in range(len(feats)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=7)

    ax.set_title('Correlation Heatmap: Student Features & Placement', fontsize=11, fontweight='bold', pad=25)
    plt.tight_layout()
    p7 = os.path.join(figures_dir, 'correlation_heatmap.png')
    plt.savefig(p7, dpi=300)
    plt.close()
    saved_plots.append(('Correlation Heatmap', p7))

    return saved_plots


def run_eda_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_data.csv')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    eda_json = os.path.join(base_dir, 'data', 'processed', 'eda_summary.json')

    headers, data, col_idx = load_cleaned_dataset(csv_path)

    print("=" * 70)
    print("[EDA STEP 1] DATASET DIMENSIONS & OVERVIEW")
    print("=" * 70)
    print(f"[*] Total Observations: {len(data)}")
    print(f"[*] Total Features: {len(headers) - 1}")

    # Descriptive statistics
    desc_stats = compute_descriptive_stats(data, col_idx)
    print("\n[EDA STEP 2] DESCRIPTIVE STATISTICS FOR NUMERICAL FEATURES")
    print("-" * 70)
    for feat, s in desc_stats.items():
        print(f"  - {feat:<22} | Mean: {s['mean']:<6} | Std: {s['std']:<6} | Median: {s['median']:<6} | Range: [{s['min']}, {s['max']}]")

    # Generate plots
    print("\n[EDA STEP 3] GENERATING & SAVING VISUALIZATION PLOTS")
    print("-" * 70)
    plots = generate_visualizations(data, col_idx, figures_dir)
    for title, path in plots:
        print(f"  [+] {title:<30} -> {path}")

    # Save summary json
    summary_data = {
        "dataset_shape": [len(data), len(headers)],
        "descriptive_stats": desc_stats,
        "placement_split": {
            "placed": sum(1 for r in data if int(r[col_idx['placement_status']]) == 1),
            "not_placed": sum(1 for r in data if int(r[col_idx['placement_status']]) == 0)
        }
    }
    with open(eda_json, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n[+] EDA summary JSON saved to: {eda_json}")
    print("=" * 70)


if __name__ == '__main__':
    run_eda_pipeline()
