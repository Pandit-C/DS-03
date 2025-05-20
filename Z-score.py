import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Button
from IPython.display import display, HTML
import io

# -------------------------------
# Dataset definition
# -------------------------------
set_a_values = [23.5, 25.1, 22.8, 24.3, 23.9, 24.7]  # Baseline
set_b_normal = [24.1, 23.8, 25.0, 24.4]              # True normal
set_b_outliers = [26.2, 27.1, 25.9, 22.3, 26.8, 21.5] # Outliers
set_b_values = set_b_normal + set_b_outliers

data = {
    'CPU': set_a_values + set_b_values,
    'Set': ['A'] * len(set_a_values) + ['B'] * len(set_b_values),
    'IsTrueOutlier': [False] * len(set_a_values) + [False] * len(set_b_normal) + [True] * len(set_b_outliers),
    'Label': [f"A{i+1}" for i in range(len(set_a_values))] + [f"B{i+1}" for i in range(len(set_b_values))]
}
df = pd.DataFrame(data)

# Z-score calculation using Set A stats
mean = df[df['Set'] == 'A']['CPU'].mean()
std = df[df['Set'] == 'A']['CPU'].std()
df['Z-Score'] = (df['CPU'] - mean) / std

export_results = []

# -------------------------------
# Main analysis function
# -------------------------------

def analyze_thresholds(threshold_min, threshold_max, target_anomalies):
    global export_results
    if threshold_min > threshold_max:
        display(HTML("<b style='color:red;'>⚠️ Min threshold must be ≤ max threshold.</b>"))
        return

    thresholds = np.arange(threshold_min, threshold_max + 0.1, 0.1)
    results = {}
    export_results = []

    for threshold in thresholds:
        detected = df[(df['Set'] == 'B') & (np.abs(df['Z-Score']) > threshold)]
        true_positives = detected[detected['IsTrueOutlier'] == True]
        false_positives = detected[detected['IsTrueOutlier'] == False]
        missed_outliers = df[(df['Set'] == 'B') & (df['IsTrueOutlier']) & (np.abs(df['Z-Score']) <= threshold)]
        results[threshold] = {
            'total': len(detected),
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'missed': len(missed_outliers),
        }
        export_results.append({
            'Threshold': threshold,
            'Detected': len(detected),
            'TruePositives': len(true_positives),
            'FalsePositives': len(false_positives),
            'MissedOutliers': len(missed_outliers),
        })

    # Summary Table
    closest_threshold = min(results.items(), key=lambda x: abs(x[1]['total'] - target_anomalies))[0]
    summary_html = "<h4>📊 Anomaly Detection Summary</h4><table><tr><th>Threshold</th><th>Total</th><th>✅ TP</th><th>❌ FP</th><th>⚠️ Missed</th></tr>"
    for t, r in results.items():
        color = 'green' if t <= 1 else 'orange' if t <= 2 else 'red'
        highlight = "font-weight:bold;" if t == closest_threshold else ""
        summary_html += f"<tr style='color:{color};{highlight}'><td>|Z| > {t:.1f}</td><td>{r['total']}</td><td>{r['true_positives']}</td><td>{r['false_positives']}</td><td>{r['missed']}</td></tr>"
    summary_html += "</table>"
    display(HTML(summary_html))

    # Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#1C2526')

    df_a = df[df['Set'] == 'A']
    df_b = df[df['Set'] == 'B']
    detected_anomalies = df_b[np.abs(df_b['Z-Score']) > threshold_min]
    missed = df_b[df_b['IsTrueOutlier'] & (np.abs(df_b['Z-Score']) <= threshold_min)]

    ax.scatter(df_a.index, df_a['CPU'], color='cyan', label='Set A (Baseline)', s=50)
    ax.scatter(df_b.index, df_b['CPU'], color='blue', label='Set B (Normal)', s=50)
    ax.scatter(detected_anomalies.index, detected_anomalies['CPU'], color='red', label='Detected Anomalies', s=100)
    ax.scatter(missed.index, missed['CPU'], color='yellow', label='Missed Outliers', s=80, edgecolors='black')

    for i, row in df.iterrows():
        ax.annotate(row['Label'], (i, row['CPU']), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='white')

    # Threshold lines
    ax.axhline(mean, color='gray', linestyle='--', label=f'Mean (μ = {mean:.1f})')
    ax.axhline(mean + threshold_min * std, color='lime', linestyle='--', label=f'+{threshold_min:.1f}σ (Min Z)')
    ax.axhline(mean - threshold_min * std, color='lime', linestyle='--')
    ax.axhline(mean + threshold_max * std, color='magenta', linestyle='--', label=f'+{threshold_max:.1f}σ (Max Z)')
    ax.axhline(mean - threshold_max * std, color='magenta', linestyle='--')

    ax.set_title("Z-Score Anomaly Detection (Set A vs Set B)", color='white', fontsize=13)
    ax.set_xlabel("Data Point Index", color='white')
    ax.set_ylabel("CPU Usage (%)", color='white')

    # ✅ Move legend outside the plot area
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor='#1C2526', labelcolor='white', frameon=True)

    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------------------
# Export Button
# -------------------------------
def export_csv_callback(button):
    if not export_results:
        display(HTML("<b style='color:red;'>⚠️ No results to export. Run the analysis first.</b>"))
        return
    df_export = pd.DataFrame(export_results)
    buf = io.StringIO()
    df_export.to_csv(buf, index=False)
    buf.seek(0)
    from google.colab import files
    files.download('anomaly_thresholds.csv')

# Button widget
export_button = Button(description='⬇️ Export Results as CSV', button_style='success')
export_button.on_click(export_csv_callback)

# Display widgets and controls
display(export_button)
interact(analyze_thresholds,
         threshold_min=FloatSlider(value=0.5, min=0.0, max=3.0, step=0.1, description='Min Z'),
         threshold_max=FloatSlider(value=3.0, min=0.0, max=3.0, step=0.1, description='Max Z'),
         target_anomalies=IntSlider(value=6, min=0, max=10, step=1, description='🎯 Target'));
