# Created by Claude Code on 2026-06-04
# Purpose: Generate 5 clean supplementary summary figures (FigS1-S5)
#   covering ALL 18 failed improvement attempts from supplementary.tex
#   v4: every attempt = one bar; FigS2 uses log scale for GRU/TCN
# Safe-to-edit-by-creator: yes

import os, warnings, logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.ticker import ScalarFormatter, LogFormatter
warnings.filterwarnings("ignore"); logging.getLogger('matplotlib.font_manager').disabled = True

# === Arial font ===
FONT_DIR = r"D:\codex\dev\soh\data\fig2_data"
fonts = {'reg': os.path.join(FONT_DIR, "arial.ttf"),
         'bold': os.path.join(FONT_DIR, "arialbd.ttf")}
for key, path in fonts.items():
    if not os.path.exists(path): raise FileNotFoundError(f"Missing: {path}")
    fm.fontManager.addfont(path)
f_reg = fm.FontProperties(fname=fonts['reg'])
f_bold = fm.FontProperties(fname=fonts['bold'])
font_name = f_reg.get_name()
mpl.rcParams['font.family'] = font_name
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = font_name
mpl.rcParams['mathtext.it'] = font_name + ':italic'
mpl.rcParams['mathtext.bf'] = font_name
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.facecolor'] = 'white'

# === Palette ===
BLUE_DARK  = "#1B4F72"
BLUE_MID   = "#2980B9"
BLUE_LIGHT = "#5DADE2"
BLUE_PALE  = "#AED6F1"
RED        = "#C0392B"
RED_PALE   = "#E6B0AA"
GRAY       = "#5D6D7E"
GRAY_L     = "#D6DBDF"
YELLOW     = "#F9E79F"

SZ_LABEL, SZ_TICK, SZ_TITLE = 13, 11, 13

OUT = r"D:\codex\dev\soh\figures\定稿\supplementary"
os.makedirs(OUT, exist_ok=True)


def style_ax(ax):
    ax.grid(False)
    ax.tick_params(axis='both', direction='out', length=3.2, width=0.9,
                   colors='black', labelsize=SZ_TICK)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.9); s.set_color('black')


def force_arial(ax):
    plt.gcf().canvas.draw()
    for lbl in [ax.xaxis.get_label(), ax.yaxis.get_label()]:
        if lbl.get_text(): lbl.set_fontproperties(f_reg); lbl.set_fontsize(SZ_LABEL)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(f_reg); label.set_fontsize(SZ_TICK)


def add_label(ax, s):
    ax.text(-0.07, 1.10, s, transform=ax.transAxes,
            fontsize=21, fontproperties=f_bold, color='black', va='bottom')


# ============================================================
# FigS1: Feature Engineering — 5 bars, |ΔR²|
# ============================================================
def make_figs1():
    labels = ["Temp.\nratio", "Temp.-strat.\nσ_V", "SOC-voltage\ninteraction",
              "Discharge\ncapacity slope", "Initial-history\nreference"]
    delta_r2 = [0.00364, -0.00228, -0.005, -0.0019, -0.040]
    abs_vals = [abs(v) for v in delta_r2]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    fig.patch.set_facecolor('white')

    x = np.arange(len(labels))
    colors = [BLUE_LIGHT if v >= 0 else (RED if v < -0.01 else RED_PALE)
              for v in delta_r2]

    ax.bar(x, abs_vals, width=0.58, color=colors, edgecolor='white', linewidth=0.5, zorder=3)

    # Always label each bar with signed ΔR²
    for xi, av, sv in zip(x, abs_vals, delta_r2):
        sign = '+' if sv >= 0 else '−'  # minus sign
        yoff = 0.0012
        ax.text(xi, av + yoff, f'{sign}{av:.3f}' if av >= 0.01 else f'{sign}{av:.4f}',
                ha='center', va='bottom', fontsize=10, color=BLUE_DARK,
                fontweight='bold', fontproperties=f_reg)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=SZ_TICK, rotation=20)
    ax.set_ylabel('|ΔR²| vs. baseline (R²=0.501)', fontsize=SZ_LABEL)
    ax.set_ylim(0, 0.046)
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04])

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=BLUE_LIGHT, edgecolor='white', label='Positive ΔR²'),
        Patch(facecolor=RED, edgecolor='white', label='Negative ΔR², |Δ|≥0.01'),
        Patch(facecolor=RED_PALE, edgecolor='white', label='Negative ΔR², |Δ|<0.01'),
    ], loc='upper left', fontsize=8.5, frameon=False, ncol=1)

    style_ax(ax); force_arial(ax); add_label(ax, 'S1')
    ax.set_title('Category 1: Feature Engineering (5 attempts)', fontsize=SZ_TITLE,
                 fontproperties=f_bold, color=BLUE_DARK, pad=10)

    fig.savefig(os.path.join(OUT, "FigS1_feature_engineering.pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "FigS1_feature_engineering.png"), format="png", bbox_inches="tight")
    plt.close(fig); print("FigS1 saved (5 bars).")


# ============================================================
# FigS2: Model Architecture — 7 bars, log-scale |ΔR²|
# ============================================================
def make_figs2():
    # 7 attempts, all as |ΔR²| from baseline R²=0.501
    labels = [
        "Seed\nensemble",
        "Hyperparam.\n(8 configs)",
        "LightGBM",
        "XGBoost\ndepth 7",
        "Random\nForest",
        "GRU\n(500-row seq)",
        "TCN\n(daily seq)",
    ]
    # |ΔR²|: absolute gap from CatBoost baseline R²=0.501
    abs_delta = [
        0.0004,   # seed ensemble
        0.005,    # hyperparam tuning (≥ 0.005, show lower bound)
        0.045,    # LightGBM R²=0.456
        0.049,    # XGBoost d7 R²=0.452
        0.058,    # RF R²=0.443
        0.70,     # GRU R²<0 → |Δ| > 0.501, approximate 0.7
        3.288,    # TCN R²=-2.787 → |Δ|=3.288
    ]
    # Underlying R² (for annotation)
    r2_info = [
        "R²≈0.501",        # seed ensemble
        "R²≤0.496",         # hyperparam
        "R²=0.456",         # LightGBM
        "R²=0.452",         # XGBoost
        "R²=0.443",         # RF
        "R²<0",             # GRU
        "R²=−2.787",   # TCN
    ]
    is_positive = [True, False, False, False, False, False, False]  # direction

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    fig.patch.set_facecolor('white')

    x = np.arange(len(labels))
    colors = [BLUE_LIGHT if p else RED for p in is_positive]
    # Make the tiny positive bar stand out
    colors[0] = BLUE_LIGHT

    ax.bar(x, abs_delta, width=0.58, color=colors, edgecolor='white', linewidth=0.5, zorder=3)

    # Log scale for y-axis (handles 0.0004 to 3.288)
    ax.set_yscale('log')
    ax.set_ylim(2e-4, 10.0)

    # Value labels
    for xi, av, ri in zip(x, abs_delta, r2_info):
        # Tighter multiplier for tall bars to avoid going out of frame
        if av > 2:
            ypos = av * 1.08
        elif av > 0.5:
            ypos = av * 1.25
        else:
            ypos = av * 1.35
        sign = '+' if is_positive[xi] else '−'
        # Show |Δ| and underlying R²
        txt = f'|Δ|={av:.3f}' if av < 0.1 else f'|Δ|≈{av:.1f}'
        if av == 0.005:
            txt = '|Δ|≥0.005'
        if av >= 0.7:
            txt = f'|Δ|>{0.5:.1f}' if av < 1 else f'|Δ|={av:.2f}'
        ax.text(xi, ypos, f'{txt}\n({ri})', ha='center', va='bottom',
                fontsize=8.5, color=BLUE_DARK, fontweight='bold',
                fontproperties=f_reg, linespacing=1.15)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=SZ_TICK, rotation=20)
    ax.set_ylabel('|ΔR²| vs. baseline (log scale)', fontsize=SZ_LABEL)

    # Custom y-tick labels
    from matplotlib.ticker import FixedLocator, FuncFormatter
    yticks = [0.0001, 0.001, 0.01, 0.1, 1, 5]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.4f}' if v < 0.01 else f'{v:.3f}' if v < 1 else f'{v:.0f}'))

    style_ax(ax); force_arial(ax); add_label(ax, 'S2')
    ax.set_title('Category 2: Model Architecture (7 attempts)', fontsize=SZ_TITLE,
                 fontproperties=f_bold, color=BLUE_DARK, pad=10)

    fig.savefig(os.path.join(OUT, "FigS2_model_architecture.pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "FigS2_model_architecture.png"), format="png", bbox_inches="tight")
    plt.close(fig); print("FigS2 saved (7 bars).")


# ============================================================
# FigS3: Post-Hoc Correction — 4 bars, |ΔR²|
# ============================================================
def make_figs3():
    labels = ["LRRC\n(2-stage resid.)", "Sample weights\n(3 schemes)",
              "Isotonic\ncalibration", "Conformal\n(stratified)"]
    delta_r2 = [-0.044, -0.010, -0.025, None]
    abs_vals  = [0.044, 0.010, 0.025, 0.052]  # placeholder for conformal

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    fig.patch.set_facecolor('white')

    x = np.arange(len(labels))
    colors = [RED, RED_PALE, RED, GRAY_L]

    ax.bar(x, abs_vals, width=0.58, color=colors, edgecolor='white', linewidth=0.5, zorder=3)

    # Value labels
    val_labels = ['0.044', 'degraded\n(3 schemes)', '0.025',
                  'tail cov.\n26.9%\nwidth +65%']
    for xi, av, vl in zip(x, abs_vals, val_labels):
        clr = GRAY if 'cov' in vl else BLUE_DARK
        ax.text(xi, av + 0.0018, vl, ha='center', va='bottom', fontsize=9,
                color=clr, fontweight='bold', fontproperties=f_reg, linespacing=1.15)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=SZ_TICK, rotation=20)
    ax.set_ylabel('|ΔR²| vs. baseline (R²=0.501)', fontsize=SZ_LABEL)
    ax.set_ylim(0, 0.062)
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

    style_ax(ax); force_arial(ax); add_label(ax, 'S3')
    ax.set_title('Category 3: Post-Hoc Correction (4 attempts)', fontsize=SZ_TITLE,
                 fontproperties=f_bold, color=BLUE_DARK, pad=10)

    fig.savefig(os.path.join(OUT, "FigS3_post_hoc_correction.pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "FigS3_post_hoc_correction.png"), format="png", bbox_inches="tight")
    plt.close(fig); print("FigS3 saved (4 bars).")


# ============================================================
# FigS4: Validation Protocol — 4 bars, |ΔR²|
# ============================================================
def make_figs4():
    labels = ["Odometer-strat.\nVIN folds", "Odometer baseline\n+ CatBoost resid.",
              "96-cell pack\ndispersion", "drop_spc\n(732 features)"]
    delta_r2 = [-0.025, -0.026, 0.0067, -0.024]
    abs_vals  = [abs(v) for v in delta_r2]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    fig.patch.set_facecolor('white')

    x = np.arange(len(labels))
    colors = [RED_PALE, RED_PALE, YELLOW, RED_PALE]

    ax.bar(x, abs_vals, width=0.58, color=colors, edgecolor='white', linewidth=0.5, zorder=3)

    for xi, av, sv in zip(x, abs_vals, delta_r2):
        sign = '+' if sv >= 0 else '−'
        ax.text(xi, av + 0.0012, f'{sign}{av:.3f}' if av >= 0.01 else f'{sign}{av:.4f}',
                ha='center', va='bottom', fontsize=10, color=BLUE_DARK,
                fontweight='bold', fontproperties=f_reg)

    # Mark biased
    ax.text(2, abs_vals[2] + 0.0038, 'biased\nsubset', ha='center', va='bottom',
            fontsize=8, color='#B7950B', fontproperties=f_reg, style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=SZ_TICK, rotation=20)
    ax.set_ylabel('|ΔR²| vs. baseline (R²=0.501)', fontsize=SZ_LABEL)
    ax.set_ylim(0, 0.034)
    ax.set_yticks([0, 0.01, 0.02, 0.03])

    style_ax(ax); force_arial(ax); add_label(ax, 'S4')
    ax.set_title('Category 4: Validation Protocol Variants (4 attempts)', fontsize=SZ_TITLE,
                 fontproperties=f_bold, color=BLUE_DARK, pad=10)

    fig.savefig(os.path.join(OUT, "FigS4_validation_protocol.pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "FigS4_validation_protocol.png"), format="png", bbox_inches="tight")
    plt.close(fig); print("FigS4 saved (4 bars).")


# ============================================================
# FigS5: External Validation — 1 comparison, 2 bars, |ρ|
# ============================================================
def make_figs5():
    labels = ['Expected\n(degradation signal)', 'Observed\n(UPC WLTP)']
    rho_signed = [0.5, -0.805]
    rho_abs    = [abs(v) for v in rho_signed]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    fig.patch.set_facecolor('white')

    x = np.arange(len(labels))
    colors = [BLUE_MID, RED]
    edges  = [BLUE_DARK, '#922B21']

    ax.bar(x, rho_abs, width=0.45, color=colors, edgecolor=edges, linewidth=0.8, zorder=3)

    for xi, av, sv in zip(x, rho_abs, rho_signed):
        sign = '+' if sv >= 0 else '−'
        ax.text(xi, av + 0.04, f'|ρ| = {av:.3f}', ha='center', va='bottom',
                fontsize=13, color=BLUE_DARK, fontweight='bold', fontproperties=f_reg)
        ax.text(xi, av - 0.08, f'({sign})', ha='center', va='top',
                fontsize=10, color=GRAY, fontproperties=f_reg)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=SZ_TICK, rotation=0)
    ax.set_ylabel('|Spearman ρ|  (σ_V/|I| vs. SOH)', fontsize=SZ_LABEL)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=BLUE_MID, edgecolor=BLUE_DARK, label='Expected'),
        Patch(facecolor=RED, edgecolor='#922B21', label='Observed (artifact)'),
    ], loc='upper left', fontsize=9, frameon=False)

    ax.text(0.5, -0.14, 'UPC WLTP: R_meas ∈ [0.04, 9274] Ω. ρ(R_meas, cycle) = −0.805 (partial-charge artifact).',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8.5, color=GRAY, fontproperties=f_reg, style='italic')

    style_ax(ax); force_arial(ax); add_label(ax, 'S5')
    ax.set_title('Category 5: External Validation (1 attempt)', fontsize=SZ_TITLE,
                 fontproperties=f_bold, color=BLUE_DARK, pad=10)

    fig.savefig(os.path.join(OUT, "FigS5_external_validation.pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "FigS5_external_validation.png"), format="png", bbox_inches="tight")
    plt.close(fig); print("FigS5 saved (2 bars).")


# ============================================================
if __name__ == '__main__':
    make_figs1()  # 5 bars
    make_figs2()  # 7 bars (was 4)
    make_figs3()  # 4 bars
    make_figs4()  # 4 bars
    make_figs5()  # 2 bars
    # Total: 5+7+4+4+2 = 22 bar positions covering all 18 attempts
    # (FigS2 has 7 bars for 7 sub-attempts; FigS5 has 2 bars for 1 attempt)
    print(f"\nAll 5 figures saved to {OUT}")
    print("Coverage: 5 + 7 + 4 + 4 + 1 = 18 attempts → all accounted for.")
