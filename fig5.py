# Fig5 v5: figura templates — #3 scatter, #1 line+shade, #9 lollipop

import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl

FIGURA_SCRIPTS = Path(r"C:\Users\yang\.claude\plugins\cache\figura\figura\0.4.0\skills\figura\scripts")
sys.path.insert(0, str(FIGURA_SCRIPTS))
import matplotlib; matplotlib.use("Agg")
import pubstyle, colors, export

ROOT = Path(r"D:\codex\dev\soh")
OUT = ROOT / "figures" / "定稿" / "fig5"
OUT.mkdir(parents=True, exist_ok=True)

pubstyle.apply()
colors.apply_cycle()
# project: 4-spine overrides
mpl.rcParams.update({
    "axes.spines.top": True, "axes.spines.right": True,
    "font.family": "Arial", "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# Fig2/3/4 palette
BLUE_DARK="#1B4F72"; BLUE_MID="#2980B9"; BLUE_LIGHT="#5DADE2"
BLUE_PALE="#AED6F1"; RED="#C0392B"; RED_BG="#F5E6E6"
GRAY="#5D6D7E"; GRAY_L="#D6DBDF"; WHITE="#FFFFFF"

def style(ax):
    ax.grid(False)
    ax.tick_params(axis="both", direction="out", length=3.2, width=0.9, colors="black", labelsize=9)
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.9); s.set_color("black")

def plabel(ax, letter):
    ax.text(-0.15, 1.05, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold", color="black")

# ---- load ----
vin_df  = pd.read_csv(ROOT/"data/fig5_data/vin_macro_lr022.csv")  # all 145 VINs, no r2 filter
tail_df = pd.read_csv(ROOT/"data/fig5_data/tail_risk_lr022.csv")
fold_df = pd.read_csv(ROOT/"data/fig4_final_perm_condition_v1_20260518/fold_metrics_full_vs_no_loadnorm_lr022_i1800.csv")
smd_df  = pd.read_csv(ROOT/"data/final_figure_data_20260514/fold1_domain_shift_smd.csv")

fold1_r2 = fold_df.iloc[0]['full_r2']
others_r2 = fold_df.iloc[1:]['full_r2'].mean()

# ---- figure: 2x2 ----
fig = plt.figure(figsize=(8.5, 6.0), dpi=300)
gs = fig.add_gridspec(2, 2, left=0.08, right=0.985, bottom=0.12, top=0.93,
                      hspace=0.52, wspace=0.40, height_ratios=[1.0, 1.0])
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# ═══ a: Template #5 box+swarm — Per-VIN MAE by label density ═══
x_a = vin_df['n'].values
y_a = vin_df['mae'].values

bins = [0, 10, 50, 100, 999]
labels = ['<10', '10-50', '50-100', '100+']
groups = [y_a[(x_a>=lo) & (x_a<hi)] for lo, hi in zip(bins[:-1], bins[1:])]
positions = np.arange(1, len(labels)+1)

# Box only
bp = ax_a.boxplot(groups, positions=positions, widths=0.45, patch_artist=True,
                   boxprops=dict(facecolor=BLUE_PALE, edgecolor=BLUE_DARK, linewidth=0.9),
                   medianprops=dict(color=BLUE_DARK, linewidth=1.4),
                   whiskerprops=dict(color=BLUE_DARK, linewidth=0.9),
                   capprops=dict(color=BLUE_DARK, linewidth=0.9),
                   flierprops=dict(marker=""))

ax_a.set_xticks(positions)
ax_a.set_xticklabels(labels, fontsize=9)
ax_a.set_xlabel('Labels per VIN', fontsize=9)
ax_a.set_ylabel('MAE (%)', fontsize=9)
ax_a.set_ylim(0, 8)
ax_a.set_yticks([0, 2, 4, 6, 8])
plabel(ax_a, "a"); style(ax_a)

# ═══ b: Template #1 line+shade — Tail SOH Bias ═══
x_b = np.arange(len(tail_df))
mae_b = tail_df['mae'].values
bias_b = tail_df['bias'].values
n_b   = tail_df['n'].values
lbl_b = tail_df['bin'].values

# MAE line — template #1
ax_b.plot(x_b, mae_b, color=BLUE_MID, lw=1.4, zorder=3)
ax_b.scatter(x_b, mae_b, s=36, c=BLUE_DARK, edgecolors=WHITE, linewidths=0.4, zorder=4)

# Bias annotations — per-point tuning
offsets = [0.8, 1.2, 1.2, 1.8, 0.9]  # tuned per point to avoid overlap
hoffsets = [0, 0, 0, 0, -0.15]  # horizontal offset for last point
for i, bias in enumerate(bias_b):
    if abs(bias) < 0.1: continue
    direction = 'over' if bias > 0 else 'under'
    color = BLUE_DARK
    ax_b.text(i + hoffsets[i], mae_b[i] + offsets[i],
              f'{bias:+.1f}\n({direction})', ha='center', va='center',
              fontsize=6.5, color=color, fontweight='bold')

# n labels
for i, n in enumerate(n_b):
    ax_b.text(i, 0.5, f'n={n}', ha='center', va='bottom', fontsize=6.5, color=GRAY)

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(lbl_b, fontsize=8)
ax_b.set_ylabel('MAE (% SOH)', fontsize=9)
ax_b.set_xlabel('SOH range', fontsize=9)
ax_b.set_ylim(0, 10)
ax_b.set_yticks([0, 2, 4, 6, 8, 10])
plabel(ax_b, "b"); style(ax_b)

# ═══ c: Template #9 lollipop — Fold1 Domain Shift ═══
smd_sorted = smd_df.sort_values('abs_smd', ascending=True)
vars_c = smd_sorted['variable'].values
smd_c  = smd_sorted['smd'].values

y_c = np.arange(len(vars_c))

# Stems — template #9 horizontal bar style
for yi, xi in zip(y_c, smd_c):
    cl = BLUE_LIGHT
    ax_c.barh(yi, xi, height=0.35, color=cl, edgecolor='none', zorder=2)
# Endpoint markers
ax_c.scatter(smd_c[smd_c>0], y_c[smd_c>0], s=28, color=BLUE_LIGHT, edgecolor=WHITE, linewidth=0.3, zorder=3)
ax_c.scatter(smd_c[smd_c<0], y_c[smd_c<0], s=28, color=BLUE_LIGHT, edgecolor=WHITE, linewidth=0.3, zorder=3)
ax_c.axvline(0, color='black', lw=0.7, zorder=1)

short = {'SOH (%)':'SOH','SOC 30d mean (%)':'SOC','Rows (90d)':'Rows',
         'Temp max 30d (C)':'Tmax','Temp min 30d (C)':'Tmin',
         'Odometer (km)':'Odo.','Days available':'Days',
         'VIN label count':'Labels','Confidence':'Conf.'}
ax_c.set_yticks(y_c)
ax_c.set_yticklabels([short.get(v,v) for v in vars_c], fontsize=7.5)
ax_c.set_xlabel('SMD (Fold1 - Others)', fontsize=9)

plabel(ax_c, "c"); style(ax_c)

# ═══ d: Per-fold R² — cross-fold generalization gap ═══
x_d = np.arange(5)
r2_d = fold_df['full_r2'].values
fold_labels = ['1', '2', '3', '4', '5']
bar_colors_d = [BLUE_LIGHT for i in range(5)]
edge_colors_d = [BLUE_DARK for i in range(5)]

ax_d.bar(x_d, r2_d, color=bar_colors_d, edgecolor=edge_colors_d, linewidth=0.7, width=0.55)

for xi, yi in zip(x_d, r2_d):
    ax_d.text(xi, yi + 0.015, f'{yi:.3f}', ha='center', va='bottom', fontsize=8, color=BLUE_DARK, fontweight='bold')

ax_d.set_xticks(x_d)
ax_d.set_xticklabels(fold_labels, fontsize=9)
ax_d.set_xlabel('Fold', fontsize=9)
ax_d.set_ylabel('R²', fontsize=9)
ax_d.set_ylim(0, 0.7)
ax_d.set_yticks([0, 0.2, 0.4, 0.6])
plabel(ax_d, "d"); style(ax_d)

for ax in [ax_a, ax_b, ax_c, ax_d]: ax.tick_params(pad=2)

export.save(fig, "Fig5_generalization_limits_v5", outdir=str(OUT), formats=("pdf", "png"))
print(f"Saved to {OUT}")
