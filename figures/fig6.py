# Fig6 v5: Reliability + External
# a=RADAR feature |rho| with error bars, b=SOH coverage, c=alpha retention-error curve
# All data from verified sources. No synthetic data.

import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl

FIGURA_SCRIPTS = Path(r"C:\Users\yang\.claude\plugins\cache\figura\figura\0.4.0\skills\figura\scripts")
sys.path.insert(0, str(FIGURA_SCRIPTS))
import matplotlib; matplotlib.use("Agg")
import pubstyle, export

ROOT = Path(r"D:\codex\dev\soh")
OUT = ROOT / "figures" / "2026-05-25" / "Fig6" / "v5"
OUT.mkdir(parents=True, exist_ok=True)

pubstyle.apply()
mpl.rcParams.update({
    "axes.spines.top": True, "axes.spines.right": True,
    "font.family": "Arial", "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

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

# ---- load conformal data ----
conf = pd.read_csv(ROOT/"data/fig6_data/conformal_oof.csv")
app = pd.read_csv(ROOT/"data/fig6_data/applicability_scores.csv")
oof = pd.read_csv(ROOT/"data/fig4_final_perm_condition_v1_20260518/aligned_oof_full_vs_no_loadnorm_lr022_i1800.csv")
bins_s = [0,80,85,90,95,120]; labels_s = ['<80','80-85','85-90','90-95','>=95']
conf['soh_bin'] = pd.cut(conf['soh'], bins=bins_s, labels=labels_s)
alpha_eval = app.merge(oof[['vin','t_s','abs_err_full']], on=['vin','t_s'], how='inner')
if len(alpha_eval) != len(app):
    raise RuntimeError(f"alpha/oof merge row mismatch: {len(alpha_eval)} vs {len(app)}")

# ---- RADAR4KIT verified values (MINIMAL_TRIAL_RESULT.md 2026-05-12) ----
features_radar = ['dv_di','sigmaV/sigmaI','V_range','V_std','V_slope','T_mean']
rho_mean = [0.674, 0.641, 0.618, 0.612, 0.329, 0.206]
rho_std  = [0.090, 0.100, 0.087, 0.090, 0.113, 0.241]
cells_sig = [12, 12, 12, 12, 12, 11]
# All verified, no synthetic values.

# ---- figure: a top full, b+c bottom ----
fig = plt.figure(figsize=(8.0, 5.5), dpi=300)
gs = fig.add_gridspec(2, 2, left=0.12, right=0.98, bottom=0.12, top=0.93,
                      hspace=0.52, wspace=0.42, height_ratios=[1.05, 1.0])
ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])

# ═══ a: RADAR4KIT per-feature |rho| + error bars ═══
y_a = np.arange(len(features_radar))[::-1]
ax_a.barh(y_a, rho_mean, xerr=rho_std, color=BLUE_LIGHT, edgecolor=BLUE_DARK,
          linewidth=0.6, height=0.55, error_kw=dict(lw=0.8, capsize=2.5, capthick=0.8, color=GRAY))
for yi, mu, sd, sig in zip(y_a, rho_mean, rho_std, cells_sig):
    ax_a.text(mu + sd + 0.03, yi, f'{mu:.3f} +/- {sd:.3f}', va='center', fontsize=7.5, color=BLUE_DARK)

ax_a.set_yticks(y_a)
ax_a.set_yticklabels(features_radar, fontsize=8)
ax_a.set_xlabel('|Spearman ρ|', fontsize=10)
ax_a.set_xlim(0, 1.05)
plabel(ax_a, "a"); style(ax_a)

# ═══ b: SOH bin → conformal coverage (line) ═══
cov_b = [conf[conf['soh_bin']==lb]['covered'].mean()*100 for lb in labels_s]
n_b   = [conf[conf['soh_bin']==lb].shape[0] for lb in labels_s]
x_b = np.arange(5)

ax_b.plot(x_b, cov_b, color=BLUE_MID, lw=1.6, marker='o', markersize=7,
          markerfacecolor=BLUE_DARK, markeredgecolor=WHITE, markeredgewidth=0.4, zorder=3)
ax_b.fill_between(x_b, cov_b, alpha=0.12, color=BLUE_LIGHT)

ax_b.axhline(88.6, color=GRAY_L, lw=0.8, ls='--')
ax_b.axhline(90, color=RED, lw=0.8, ls=':')
ax_b.text(3.8, 88.3, 'pooled', fontsize=7, color=GRAY_L, ha='right')
ax_b.text(3.8, 90.3, 'target 90%', fontsize=7, color=RED, ha='right')

for xi, cv, ni in zip(x_b, cov_b, n_b):
    if xi < 4:
        ax_b.text(xi, cv + 3.5, f'{cv:.0f}%', ha='center', va='bottom', fontsize=8,
                  color=BLUE_DARK, fontweight='bold')
    else:
        ax_b.text(xi + 0.05, cv + 3.5, f'{cv:.0f}%', ha='center', va='bottom', fontsize=8,
                  color=BLUE_DARK, fontweight='bold')
    nx = xi if xi < 4 else 3.6
    ax_b.text(nx + 0.08, 2, f'n={ni}', ha='center', va='bottom', fontsize=6.5, color=GRAY)

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(labels_s, fontsize=8)
ax_b.set_ylabel('Coverage (%)', fontsize=10)
ax_b.set_xlabel('SOH range (%)', fontsize=10)
ax_b.set_ylim(0, 105)
plabel(ax_b, "b"); style(ax_b)

# ═══ c: alpha applicability → retention-error curve ═══
retentions = np.array([100, 90, 80, 70, 60, 50])
alpha_sorted = alpha_eval.sort_values('alpha_no_s4', ascending=False).reset_index(drop=True)
mae_c = []
n_c = []
for keep in retentions:
    k = int(np.ceil(len(alpha_sorted) * keep / 100.0))
    subset = alpha_sorted.iloc[:k]
    mae_c.append(subset['abs_err_full'].mean())
    n_c.append(k)
mae_c = np.array(mae_c)
x_c = np.arange(len(retentions))

ax_c.plot(x_c, mae_c, color=BLUE_MID, lw=1.6, marker='o', markersize=7,
          markerfacecolor=BLUE_DARK, markeredgecolor=WHITE, markeredgewidth=0.4, zorder=3)
ax_c.fill_between(x_c, mae_c, alpha=0.12, color=BLUE_LIGHT)

for xi, keep, mae, ni in zip(x_c, retentions, mae_c, n_c):
    ax_c.text(xi, mae + 0.035, f'{mae:.2f}', ha='center', va='bottom', fontsize=8,
              color=BLUE_DARK, fontweight='bold')
    ax_c.text(xi, 1.34, f'n={ni}', ha='center', va='bottom', fontsize=6.2, color=GRAY)

delta90 = mae_c[1] - mae_c[0]
ax_c.annotate(f'90%: {delta90:+.2f}', xy=(1, mae_c[1]), xytext=(1.25, mae_c[1] + 0.20),
              arrowprops=dict(arrowstyle='->', color=RED, lw=0.8),
              fontsize=7, color=RED, ha='left')

ax_c.set_xticks(x_c)
ax_c.set_xticklabels([f'{x}%' for x in retentions], fontsize=8)
ax_c.set_ylabel('MAE (% SOH)', fontsize=10)
ax_c.set_xlabel('Retained predictions by α (%)', fontsize=10)
ax_c.set_ylim(1.30, max(mae_c) + 0.35)
plabel(ax_c, "c"); style(ax_c)

for ax in [ax_a, ax_b, ax_c]: ax.tick_params(pad=2)

export.save(fig, "Fig6_reliability_v5", outdir=str(OUT), formats=("pdf", "png"))

(OUT/"Fig6_v5_manifest.json").write_text(json.dumps({
    "figure": "Fig6_reliability_v5",
    "layout": "a top full, b+c bottom",
    "panels": {
        "a": "RADAR4KIT: per-feature |Spearman rho| with +/-1 std error bars. dv_di=0.67 strongest.",
        "b": "Conformal SOH coverage. Pooled 88.6%, <80% only 29%. Tail failure.",
        "c": "Alpha applicability retention-error curve. Higher alpha retained subset has lower MAE; weak-to-moderate monotonic risk stratification.",
    },
    "data": {
        "RADAR4KIT": "MINIMAL_TRIAL_RESULT.md (verified 2026-05-12). 12 cells, mean+/-std. No synthetic.",
        "conformal": "reliability_module3_v1_20260509/conformal_oof.csv (9619 samples). Verified.",
        "applicability": "fig6_data/applicability_scores.csv merged with final lr022 aligned OOF abs_err_full. No synthetic.",
    },
    "alpha_retention": {
        "retained_percent": retentions.tolist(),
        "mae_percent_soh": [float(x) for x in mae_c],
        "n": [int(x) for x in n_c],
        "note": "Sorted by alpha_no_s4 descending; y is mean abs_err_full from final lr022_i1800 OOF."
    },
    "palette": "Fig2/3/4/5 unified blue",
}, indent=2), encoding="utf-8")
