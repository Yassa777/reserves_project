"""
Create publication-quality figures for full 55-combination model matrix.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

OUTPUT_DIR = Path("/Users/dim/Desktop/Spill Projects/SL-FSI/reserves_project/academic_deliverables")
FIGURES_DIR = OUTPUT_DIR / "figures"

def create_heatmap():
    """Create heatmap of RMSE vs Naive for all combinations."""
    pivot = pd.read_csv(OUTPUT_DIR / "full_70_pivot.csv", index_col=0)

    # Reorder columns
    col_order = ['parsimonious', 'bop', 'monetary', 'pca', 'full']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    # Sort rows by average performance
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg')
    pivot = pivot.drop('avg', axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap data
    data = pivot.values
    models = pivot.index.tolist()
    varsets = [c.upper()[:4] for c in pivot.columns]

    # Color map: green for negative (beats naive), red for positive
    cmap = plt.cm.RdYlGn_r
    vmax = min(200, np.nanmax(np.abs(data[np.abs(data) < 500])))

    # Clip extreme values for visualization
    data_clipped = np.clip(data, -100, 200)

    im = ax.imshow(data_clipped, cmap=cmap, aspect='auto', vmin=-70, vmax=200)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('RMSE vs Naive (%)', fontsize=12)

    # Labels
    ax.set_xticks(range(len(varsets)))
    ax.set_xticklabels(varsets, fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(varsets)):
            val = data[i, j]
            if np.isnan(val):
                text = 'N/A'
                color = 'gray'
            elif val > 500:
                text = f'{val:.0f}%'
                color = 'white'
            else:
                text = f'{val:+.1f}%'
                color = 'white' if abs(val) > 50 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9, fontweight='bold')

    ax.set_xlabel('Variable Set', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Matrix: RMSE vs Naive Benchmark (%)\nPost-Crisis Period (2024-07+) | Green = Better, Red = Worse',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table2_full_heatmap.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'table2_full_heatmap.png'}")

def create_detailed_table():
    """Create detailed metrics table figure."""
    df = pd.read_csv(OUTPUT_DIR / "full_70_combinations.csv")
    df = df[df['status'] == 'ok'].copy()

    # Sort by RMSE within each varset
    df = df.sort_values(['variable_set', 'rmse'])

    fig, ax = plt.subplots(figsize=(18, 16))
    ax.axis('off')

    headers = ['VarSet', 'Model', 'N', 'RMSE', 'vs Naive', 'MAPE', 'sMAPE', 'R²', 'MASE', 'U2', 'Dir%', 'Beats']

    table_data = []
    for _, row in df.iterrows():
        beats = '✓' if row['beats_naive'] else ''
        r2 = f"{row['r_squared']:.3f}" if row['r_squared'] > -10 else '<-10'
        table_data.append([
            row['variable_set'][:4].upper(),
            row['model'][:10],
            int(row['n']),
            f"{row['rmse']:.0f}",
            f"{row['rmse_vs_naive_pct']:+.1f}%",
            f"{row['mape']:.1f}%",
            f"{row['smape']:.1f}%",
            r2,
            f"{row['mase']:.2f}",
            f"{row['theil_u2']:.2f}",
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
    table.set_fontsize(8)
    table.scale(1.1, 1.3)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Color cells based on performance
    for i, row_data in enumerate(table_data, 1):
        # Color vs naive column
        try:
            val = float(row_data[4].replace('%', '').replace('+', ''))
            if val < -10:
                table[(i, 4)].set_facecolor('#C6EFCE')  # Green
            elif val < 0:
                table[(i, 4)].set_facecolor('#FFEB9C')  # Yellow
            elif val < 50:
                table[(i, 4)].set_facecolor('#FFC7CE')  # Light red
            else:
                table[(i, 4)].set_facecolor('#FF6666')  # Dark red
        except:
            pass

        # Color beats column
        if row_data[11] == '✓':
            table[(i, 11)].set_facecolor('#C6EFCE')
            table[(i, 11)].set_text_props(weight='bold', color='green')

    plt.title('Table 3: Complete Model × Variable Set Results (55 Combinations)\nPost-Crisis Period (2024-07+)',
              fontsize=14, weight='bold', pad=20)

    # Footnote
    footnote = """
    Notes: RMSE = Root Mean Square Error (USD M), MAPE = Mean Absolute Percentage Error, sMAPE = Symmetric MAPE,
    R² = Coefficient of Determination, MASE = Mean Absolute Scaled Error, U2 = Theil U2, Dir% = Directional Accuracy.
    Green highlighting indicates model beats naive benchmark. Values clipped for display.
    """
    plt.figtext(0.5, 0.01, footnote, ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table3_full_detailed.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'table3_full_detailed.png'}")

def create_winner_summary():
    """Create summary figure for winning models."""
    df = pd.read_csv(OUTPUT_DIR / "full_70_combinations.csv")
    winners = df[df['beats_naive'] == True].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    headers = ['Rank', 'Model', 'VarSet', 'RMSE', 'vs Naive', 'MAPE', 'Dir Acc']

    winners = winners.sort_values('rmse_vs_naive_pct')
    table_data = []
    for i, (_, row) in enumerate(winners.iterrows(), 1):
        table_data.append([
            i,
            row['model'],
            row['variable_set'].upper(),
            f"{row['rmse']:.1f}",
            f"{row['rmse_vs_naive_pct']:+.1f}%",
            f"{row['mape']:.1f}%",
            f"{row['dir_acc']:.0f}%",
        ])

    if not table_data:
        table_data = [['No models beat naive', '', '', '', '', '', '']]

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E7D32')
        table[(0, i)].set_text_props(color='white', weight='bold')

    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor('#C8E6C9')

    plt.title('Models That Beat Naive Benchmark\nPost-Crisis Period (2024-07+)',
              fontsize=14, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table_winners.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'table_winners.png'}")

def main():
    print("Creating full matrix figures...")
    create_heatmap()
    create_detailed_table()
    create_winner_summary()
    print("\nDone!")

if __name__ == "__main__":
    main()
