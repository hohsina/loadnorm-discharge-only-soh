# Fig3 v5: EXACT template replica, lr022_i1800 OOF CSV data
# Layout: scatter + CDF (top row), violin (bottom full-row)
# Dimensions: 7.16 x 7.16 inch square

import os, sys, warnings, logging
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager as fm
warnings.filterwarnings("ignore"); logging.getLogger('matplotlib.font_manager').disabled = True

# === VERBATIM Arial font engine ===
FONT_DIR = "/root/autodl-tmp/results1"
fonts = {'reg':os.path.join(FONT_DIR,"arial.ttf"),'bold':os.path.join(FONT_DIR,"arialbd.ttf"),
         'ital':os.path.join(FONT_DIR,"ariali.ttf"),'bi':os.path.join(FONT_DIR,"arialbi.ttf")}
for key, path in fonts.items():
    if not os.path.exists(path): raise FileNotFoundError(f"Missing: {path}")
    fm.fontManager.addfont(path)
f_reg = fm.FontProperties(fname=fonts['reg'])
font_name = f_reg.get_name()
mpl.rcParams['font.family'] = font_name
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = font_name
mpl.rcParams['mathtext.it'] = font_name+':italic'
mpl.rcParams['mathtext.bf'] = font_name
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.facecolor'] = 'white'
SIZE_LABEL, SIZE_TICK = 12, 10

def force_ari_font_all(ax):
    plt.gcf().canvas.draw()
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set_fontproperties(f_reg); ax.xaxis.label.set_fontsize(SIZE_LABEL); ax.xaxis.label.set_color('black')
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set_fontproperties(f_reg); ax.yaxis.label.set_fontsize(SIZE_LABEL); ax.yaxis.label.set_color('black')
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontproperties(f_reg); label.set_fontsize(SIZE_TICK); label.set_color('black')
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts(): text.set_fontproperties(f_reg); text.set_fontsize(SIZE_TICK)
    for artist in ax.get_children():
        if isinstance(artist, plt.Text): artist.set_fontproperties(f_reg); artist.set_fontsize(SIZE_TICK)

# === Load lr022_i1800 OOF data (REPLACES old .npy files) ===
OOF = "/root/autodl-tmp/results1_derived/audit_breakthrough_v2_20260511/breakthrough_audit_v2_lr022_i1800_oof.csv"
df = pd.read_csv(OOF)
all_gts = df["y_true"].values; all_preds = df["y_pred"].values
errors = all_preds - all_gts; all_abs_err = np.abs(errors)
overall_rmse = np.sqrt(mean_squared_error(all_gts, all_preds))
print(f"Loaded {len(df)} OOF rows, RMSE={overall_rmse:.2f}")

OUT = sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/results1/fig3_perf_v5_20260515"
os.makedirs(OUT,exist_ok=True)

# === FIGURE: EXACT template layout (7.16x7.16, 2x2 GridSpec) ===
fig = plt.figure(figsize=(7.16, 7.16), dpi=300)
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])   # scatter
ax2 = fig.add_subplot(gs[0, 1])   # CDF
ax3 = fig.add_subplot(gs[1, :])   # violin (full bottom row)

# ===== Panel A: Scatter =====
xlim_range = (70, 105)
ax1.scatter(all_gts, all_preds, alpha=0.6, s=5, c='#3498DB', marker='o',
            edgecolors='white', linewidths=0.2, zorder=2, label='Test Samples')
ax1.plot(xlim_range, xlim_range, color='#C0392B', linestyle='--', linewidth=1.5, zorder=3, label='Ideal Fit')
ax1.set_xlim(xlim_range); ax1.set_ylim(xlim_range)
ax1.set_xticks(np.arange(70, 106, 10)); ax1.set_yticks(np.arange(70, 106, 10))
ax1.set_xlabel('True SOH (%)'); ax1.set_ylabel('Predicted SOH (%)')
ax1.legend(loc='upper left', frameon=False, prop=f_reg)

# ===== Panel B: Error CDF =====
sorted_errors = np.sort(all_abs_err)
cdf_y = np.arange(1, len(sorted_errors)+1)/len(sorted_errors)*100
ax2.plot(sorted_errors, cdf_y, color='#2980B9', linewidth=2.0, zorder=3)
ax2.fill_between(sorted_errors, 0, cdf_y, color='#AED6F1', alpha=0.3, zorder=2)
p80_err = np.percentile(all_abs_err, 80); p90_err = np.percentile(all_abs_err, 90)
ax2.axvline(x=p90_err, ymax=0.9, color='#E74C3C', linestyle='--', linewidth=1.2, zorder=4)
ax2.axhline(y=90, xmax=p90_err/max(10, p90_err*1.5), color='#E74C3C', linestyle='--', linewidth=1.2, zorder=4)
ax2.text(p90_err+0.7, 82, f'90% samples\n< {p90_err:.2f}%', color='#C0392B')
ax2.scatter([p90_err], [90], color='#C0392B', s=25, zorder=5)
ax2.axvline(x=p80_err, ymax=0.8, color='#27AE60', linestyle='-.', linewidth=1.2, zorder=4)
ax2.axhline(y=80, xmax=p80_err/max(10, p90_err*1.5), color='#27AE60', linestyle='-.', linewidth=1.2, zorder=4)
ax2.text(p80_err+0.45, 70, f'80% samples\n< {p80_err:.2f}%', color='#1E8449')
ax2.scatter([p80_err], [80], color='#1E8449', s=25, zorder=5)
ax2.set_xlim(0, max(10, p90_err*1.5)); ax2.set_ylim(0, 105)
ax2.set_xlabel('Absolute Error (%)'); ax2.set_ylabel('Cumulative Probability (%)')
ax2.grid(False)

# ===== Panel C: Violin by SOH stage =====
stage1_err = all_abs_err[all_gts >= 95]
stage2_err = all_abs_err[(all_gts >= 90) & (all_gts < 95)]
stage3_err = all_abs_err[(all_gts >= 85) & (all_gts < 90)]
stage4_err = all_abs_err[all_gts < 85]
data_to_plot, labels = [], []
if len(stage4_err) > 0: data_to_plot.append(stage4_err); labels.append('< 85')
if len(stage3_err) > 0: data_to_plot.append(stage3_err); labels.append('[85, 90)')
if len(stage2_err) > 0: data_to_plot.append(stage2_err); labels.append('[90, 95)')
if len(stage1_err) > 0: data_to_plot.append(stage1_err); labels.append('[95, 100]')

if data_to_plot:
    parts = ax3.violinplot(data_to_plot, showmeans=False, showmedians=False, showextrema=False)
    colors = ['#F5CBA7','#F9E79F','#A9DFBF','#AED6F1'][:len(data_to_plot)]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color); pc.set_edgecolor('black'); pc.set_alpha(0.7); pc.set_linewidth(0.8)
    bplot = ax3.boxplot(data_to_plot, patch_artist=True, widths=0.15,
        medianprops=dict(color='#C0392B', linewidth=1.5),
        flierprops=dict(marker='o', color='#95A5A6', markersize=1.5, alpha=0.3), zorder=3)
    for patch in bplot['boxes']:
        patch.set_facecolor('white'); patch.set_edgecolor('#34495E'); patch.set_linewidth(1.0)
    ax3.set_xticks(np.arange(1, len(labels)+1))
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Absolute Error (%)')
    ax3.set_xlabel('True SOH Interval (%)')
    ax3.set_ylim(0, 16)
    ax3.set_yticks(np.arange(0, 17, 4))

# === VERBATIM axis styling ===
for ax in [ax1, ax2, ax3]:
    ax.grid(False)
    ax.tick_params(axis='x', which='major', direction='out', bottom=True, top=False, length=4, width=1.0)
    ax.tick_params(axis='y', which='major', direction='out', left=True, right=False, length=4, width=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color('black')
    force_ari_font_all(ax)

fig.savefig(os.path.join(OUT,"Fig3_main_performance.pdf"),format="pdf",bbox_inches="tight")
fig.savefig(os.path.join(OUT,"Fig3_main_performance.png"),format="png",bbox_inches="tight")
print(f"Fig3 v5 saved to {OUT}")
