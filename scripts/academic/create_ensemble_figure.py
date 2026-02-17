"""Create publication-quality figure for ensemble results."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

OUTPUT_DIR = Path("/Users/dim/Desktop/Spill Projects/SL-FSI/reserves_project/academic_deliverables")
FIGURES_DIR = OUTPUT_DIR / "figures"

def main():
    df = pd.read_csv(OUTPUT_DIR / "ensemble_results.csv")
    df = df.sort_values('rmse')

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['green' if v < 0 else 'salmon' for v in df['vs_naive']]

    bars = ax.barh(range(len(df)), df['vs_naive'], color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['method'], fontsize=11)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Naive Benchmark')
    ax.set_xlabel('RMSE vs Naive (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble Methods Performance\nPost-Crisis Period (2024-07+) | Green = Beats Naive',
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['vs_naive'] + 2, i, f"{row['vs_naive']:+.1f}%", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ensemble_comparison.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'ensemble_comparison.png'}")

    # Create detailed table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    headers = ['Method', 'N', 'RMSE', 'vs Naive', 'MAPE', 'Dir Acc', 'Beats']

    table_data = []
    for _, row in df.iterrows():
        beats = '✓' if row['vs_naive'] < 0 else ''
        table_data.append([
            row['method'],
            int(row['n']),
            f"{row['rmse']:.1f}",
            f"{row['vs_naive']:+.1f}%",
            f"{row['mape']:.1f}%",
            f"{row['dir_acc']:.0f}%",
            beats,
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    for i, row_data in enumerate(table_data, 1):
        if row_data[6] == '✓':
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#C6EFCE')

    plt.title('Ensemble Stacking Methods - Detailed Results', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ensemble_table.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'ensemble_table.png'}")

if __name__ == "__main__":
    main()
