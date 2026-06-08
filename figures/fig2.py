# Fig2 v6: small correction of final Fig2 data口径; same layout/style; C label spacing fixed.
import os, sys, warnings, logging, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
warnings.filterwarnings("ignore"); logging.getLogger('matplotlib.font_manager').disabled = True

# === VERBATIM Arial registration ===
FONT_DIR = "/root/autodl-tmp/results1"
fonts = {'reg':os.path.join(FONT_DIR,"arial.ttf"),'bold':os.path.join(FONT_DIR,"arialbd.ttf"),
         'ital':os.path.join(FONT_DIR,"ariali.ttf"),'bi':os.path.join(FONT_DIR,"arialbi.ttf")}
for key, path in fonts.items():
    if not os.path.exists(path): raise FileNotFoundError(f"Missing: {path}")
    fm.fontManager.addfont(path)
f_reg = fm.FontProperties(fname=fonts['reg'])
f_bold = fm.FontProperties(fname=fonts['bold'])
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
    if ax.xaxis.label.get_text(): ax.xaxis.label.set_fontproperties(f_reg); ax.xaxis.label.set_fontsize(SIZE_LABEL)
    if ax.yaxis.label.get_text(): ax.yaxis.label.set_fontproperties(f_reg); ax.yaxis.label.set_fontsize(SIZE_LABEL)
    for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontproperties(f_reg); label.set_fontsize(SIZE_TICK)
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts(): text.set_fontproperties(f_reg); text.set_fontsize(SIZE_TICK)
    for artist in ax.get_children():
        if isinstance(artist, plt.Text): artist.set_fontproperties(f_reg); artist.set_fontsize(SIZE_TICK)

# === NEW data sources (REPLACES old parquet1 + soh_labels_*.csv) ===
FEAT = "/root/autodl-tmp/results1/le20_conf_ge_0p55_features_loadnorm_20260502_2324/le20_conf_ge_0p55_loadnorm.csv"
PAIRS = "/root/autodl-tmp/results1/final_figure_data_20260514/repeatability_pairs.csv"
OUT = sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/results1/fig2_v6_20260518"
os.makedirs(OUT,exist_ok=True)

df = pd.read_csv(FEAT)
df["soh"] = pd.to_numeric(df["soh"], errors="coerce")
df = df.dropna(subset=["vin","soh"]).reset_index(drop=True)
pairs = pd.read_csv(PAIRS)
odo = pd.to_numeric(df.get("odometer_before_label_km", np.nan), errors="coerce")
print(f"Loaded: {len(df)} labels, {df.vin.nunique()} VINs, {len(pairs)} pairs")

# Use frozen repeatability tier from repeatability_pairs.csv.
print("Pair confidence tiers:", pairs["pair_confidence_tier"].dropna().unique().tolist())

# === FIGURE: EXACT old Fig2 layout (12x8, 2x3 GridSpec) ===
fig = plt.figure(figsize=(12, 8), dpi=300)
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.30, height_ratios=[1,1], width_ratios=[1,1,1])
ax_a = fig.add_subplot(gs[0, :])   # top row full width
ax_b = fig.add_subplot(gs[1, 0])   # bottom left
ax_c = fig.add_subplot(gs[1, 1])   # bottom center
ax_d = fig.add_subplot(gs[1, 2])   # bottom right

# ===== Panel A: SOH vs Odometer scatter (replaces Cycle Number) =====
mask = odo.between(0,400000) & df["soh"].between(75,100)
sample = df.loc[mask].sample(min(8000, mask.sum()), random_state=20260514)
ax_a.scatter(sample["odometer_before_label_km"], sample["soh"], s=4, color='#2980B9',
             alpha=0.6, edgecolors='none', marker='o', zorder=2)
# Red dashed trend line
odo_v = odo[mask].dropna().values; soh_v = df["soh"][mask].dropna().values
if len(odo_v) > 1:
    m, b = np.polyfit(odo_v, soh_v, 1)
    xl = np.linspace(odo_v.min(), odo_v.max(), 100)
    ax_a.plot(xl, m*xl+b, color='#C0392B', linestyle='--', linewidth=1.5, zorder=3)
    rho, _ = spearmanr(odo_v, soh_v)
    ax_a.text(0.985, 0.94, f"Spearman ρ={rho:.3f}\nn={len(df):,}, VINs={df.vin.nunique()}",
              transform=ax_a.transAxes, ha="right", va="top",
              fontproperties=f_reg, fontsize=SIZE_TICK, color="#1B4F72")

pass  # no legend
ax_a.set_xlabel('Odometer (×10³ km)', fontsize=SIZE_LABEL); ax_a.set_ylabel('Proxy SOH (%)', fontsize=SIZE_LABEL)
ax_a.set_ylim(75, 100); ax_a.set_yticks(np.arange(75, 101, 5))
ax_a.set_xlim(-5000, 410000)
ax_a.xaxis.set_major_locator(plt.MaxNLocator(16, integer=True))
ax_a.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1000:.0f}'))
ax_a.yaxis.set_major_locator(plt.MaxNLocator(6, integer=True))

# ===== Panel B: Repeatability histogram (replaces SOC KDE) =====
# Signed ΔSOH — symmetric distribution proves no systematic bias
deltas = pairs["soh_2"].values - pairs["soh_1"].values
# Clip to ±3σ for display
s = np.std(deltas); lo, hi = max(deltas.min(), -4*s), min(deltas.max(), 4*s)
sns.kdeplot(deltas, ax=ax_b, color='#1B4F72', linewidth=2.0, zorder=3)
xs, ys = ax_b.lines[0].get_xdata(), ax_b.lines[0].get_ydata()
y_peak = np.max(ys); ys_rel = (ys/y_peak*100.0) if y_peak > 0 else ys
ax_b.lines[0].set_ydata(ys_rel)
ax_b.fill_between(xs, 0, ys_rel, color='#5DADE2', alpha=0.3, zorder=2)

q1, median, q3 = np.percentile(deltas, [25,50,75])
p1, p99 = np.percentile(deltas, [1,99])
ax_b.axvspan(q1, q3, color='#AED6F1', alpha=0.4, zorder=1)
ax_b.axvline(0, color='#7f8c8d', linestyle='-', linewidth=0.8, zorder=1)
ax_b.axvline(median, color='#C0392B', linestyle='--', linewidth=1.5, zorder=4)
mad_abs = pairs["abs_delta_soh"].median()
p95_abs = pairs["abs_delta_soh"].quantile(0.95)
ax_b.text(0.04, 0.92, f"MAD={mad_abs:.2f} pp\nP95={p95_abs:.2f} pp",
          transform=ax_b.transAxes, ha="left", va="top",
          fontproperties=f_reg, fontsize=SIZE_TICK, color="#1B4F72")
ax_b.set_xlabel('ΔSOH (%)', fontsize=SIZE_LABEL)
ax_b.set_ylabel('Probability (%)', fontsize=SIZE_LABEL)
ax_b.set_xlim(-6, 6); ax_b.set_xticks(np.arange(-6, 7, 2))
ax_b.set_ylim(0, 100)

# ===== Panel C: Confidence tier vs repeatability (frozen 3-tier口径) =====
tier_order = ["medium", "high", "very_high"]
tier_labels = ["Medium", "High", "Very High"]
mad_vals = []
n_vals = []
for tier in tier_order:
    grp = pairs[pairs["pair_confidence_tier"] == tier]
    mad_vals.append(grp["abs_delta_soh"].median())
    n_vals.append(len(grp))
colors_c = ['#5DADE2', '#4292C6', '#2B6BAE']
ax_c.bar(range(3), mad_vals, color=colors_c, edgecolor='none', width=0.55, zorder=3)
ax_c.set_xticks(range(3))
ax_c.set_xticklabels(tier_labels, fontsize=SIZE_TICK)
ax_c.set_xlabel('Label quality', fontsize=SIZE_LABEL-1)
ax_c.set_ylabel('Repeatability MAD (%)', fontsize=SIZE_LABEL)
ax_c.set_ylim(0, 1.45)
ax_c.set_yticks(np.arange(0, 1.46, 0.3))
ax_c.set_xlim(-0.5, 2.5)
for i, (mad, n) in enumerate(zip(mad_vals, n_vals)):
    ax_c.text(i, mad + 0.04, f"{mad:.2f}\n(n={n:,})", ha="center", va="bottom",
              fontproperties=f_reg, fontsize=7.2, color="#1B4F72")

# ===== Panel D: Odometer coverage (KDE) =====
odo_d = pd.to_numeric(df["odometer_before_label_km"], errors="coerce").dropna()
sns.kdeplot(odo_d, ax=ax_d, color='#1B4F72', linewidth=2.0, zorder=3)
xs, ys = ax_d.lines[0].get_xdata(), ax_d.lines[0].get_ydata()
ys_scaled = ys * 1e6
ax_d.lines[0].set_ydata(ys_scaled)
ax_d.fill_between(xs, 0, ys_scaled, color='#5DADE2', alpha=0.3, zorder=2)
q1, med, q3 = np.percentile(odo_d, [25, 50, 75])
ax_d.axvspan(q1, q3, color='#AED6F1', alpha=0.4, zorder=1)
ax_d.axvline(med, color='#C0392B', linestyle='--', linewidth=1.5, zorder=4)
ax_d.set_xlabel('Odometer (×10³ km)', fontsize=SIZE_LABEL)
ax_d.set_ylabel('Density', fontsize=SIZE_LABEL)
ax_d.set_xlim(-5000, 420000)
ax_d.xaxis.set_major_locator(plt.MaxNLocator(6, integer=True))
ax_d.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1000:.0f}'))
ax_d.yaxis.set_major_locator(plt.MaxNLocator(5))

# === VERBATIM axis styling ===
for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.grid(False)
    ax.tick_params(axis='x', which='major', direction='out', bottom=True, top=False, length=4, width=1.0)
    ax.tick_params(axis='y', which='major', direction='out', left=True, right=False, length=4, width=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color('black')
    force_ari_font_all(ax)

# Panel C: suppress x tick marks (labels only)
ax_c.tick_params(axis='x', length=0)

fig.savefig(os.path.join(OUT,"Fig2_label_validity_v6.pdf"),format="pdf",bbox_inches="tight")
fig.savefig(os.path.join(OUT,"Fig2_label_validity_v6.png"),format="png",bbox_inches="tight")
print(f"Fig2 v6 saved to {OUT}")
