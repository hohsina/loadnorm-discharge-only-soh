# v30: a=KDE full top, b=ablation bar bottom-left, c=3x4 heatmap bottom-right
# Current re-binned to quartiles for uniform 4-column layout

import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import gaussian_kde

FIGURA_SCRIPTS = Path(r"C:\Users\yang\.claude\plugins\cache\figura\figura\0.4.0\skills\figura\scripts")
sys.path.insert(0, str(FIGURA_SCRIPTS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
import pubstyle, export

ROOT = Path(r"D:\codex\dev\soh")
DATA = ROOT / "data" / "fig4_final_perm_condition_v1_20260518"
OUT = ROOT / "figures" / "2026-05-16" / "Fig4" / "v30"
OUT.mkdir(parents=True, exist_ok=True)

pubstyle.apply()
mpl.rcParams.update({
    "axes.spines.top": True, "axes.spines.right": True,
    "font.family": "Arial", "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

BLUE_DARK="#1B4F72"; BLUE_MID="#2980B9"; BLUE_BRIGHT="#3498DB"
BLUE_LIGHT="#5DADE2"; BLUE_PALE="#AED6F1"; BLUE_GHOST="#D6EAF8"
RED="#C0392B"; GRAY="#5D6D7E"; GRAY_L="#D6DBDF"; WHITE="#FFFFFF"

def style(ax):
    ax.grid(False)
    ax.tick_params(axis="both", direction="out", length=3.2, width=0.9, colors="black", labelsize=9)
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.9); s.set_color("black")

def plabel(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=14, fontweight="bold", color="black")

# ---- load data ----
cond = pd.read_csv(DATA / "condition_controlled_full_vs_no_loadnorm.csv")
corr = pd.read_csv(DATA / "ln_physical_corr_v1.csv")
oof  = pd.read_csv(DATA / "aligned_oof_full_vs_no_loadnorm_lr022_i1800.csv")
summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))

full_r2 = summary["full_metrics"]["r2"]; no_ln_r2 = summary["no_loadnorm_metrics"]["r2"]
fixed_gain = summary["delta"]["r2_full_minus_no_loadnorm"]
context_r2 = 0.252; voltage_r2 = 0.299

# ---- figure: 8x5.5, 2-row ----
fig = plt.figure(figsize=(8.0, 5.5), dpi=300)
gs = fig.add_gridspec(2, 2, left=0.08, right=0.985, bottom=0.12, top=0.955,
                      hspace=0.45, wspace=0.38, height_ratios=[1.05, 1.0])
ax_a = fig.add_subplot(gs[0, :])  # top row full = KDE
ax_b = fig.add_subplot(gs[1, 0])  # bottom left = ablation bar
ax_c = fig.add_subplot(gs[1, 1])  # bottom right = heatmap

# ============ Panel a: KDE |r_SOH| ============
ln_r = corr[corr["is_loadnorm"]]["r_soh"].abs().dropna().values
nl_r = corr[~corr["is_loadnorm"]]["r_soh"].abs().dropna().values
xs = np.linspace(0, 0.55, 200)
kde_ln = gaussian_kde(ln_r)(xs); kde_nl = gaussian_kde(nl_r)(xs)
ax_a.fill_between(xs, kde_ln, alpha=0.40, color=BLUE_MID, lw=0)
ax_a.fill_between(xs, kde_nl, alpha=0.25, color=BLUE_PALE, lw=0)
ax_a.plot(xs, kde_ln, color=BLUE_DARK, lw=1.8, label="LoadNorm features")
ax_a.plot(xs, kde_nl, color=BLUE_LIGHT, lw=1.5, ls="--", label="Non-LN features")
med_ln=np.median(ln_r); med_nl=np.median(nl_r)
ax_a.axvline(med_ln, color=BLUE_DARK, lw=1.2, ls=":")
ax_a.axvline(med_nl, color=BLUE_LIGHT, lw=1.2, ls=":")
ax_a.text(med_ln + 0.002, ax_a.get_ylim()[1] * 0.86, "median",
          ha="left", va="top", fontsize=7, color=BLUE_DARK)
ax_a.text(med_nl + 0.002, ax_a.get_ylim()[1] * 0.86, "median",
          ha="left", va="top", fontsize=7, color=BLUE_LIGHT)
ax_a.legend(loc="upper right",fontsize=9,frameon=True,facecolor=WHITE,edgecolor=GRAY_L,framealpha=0.9,
             bbox_to_anchor=(0.97,0.97))
ax_a.set_xlabel("|Spearman r|",fontsize=11)
ax_a.set_ylabel("Density",fontsize=11)
ax_a.yaxis.set_label_position("left")
ax_a.set_xlim(0,0.55); ax_a.set_ylim(0,None)
plabel(ax_a,"a"); style(ax_a)

# ============ Panel b: Ablation bar ============
tags=["Context","Voltage","No LN","Full"]
r2s=[context_r2,voltage_r2,no_ln_r2,full_r2]
bar_c=[BLUE_PALE,BLUE_LIGHT,BLUE_BRIGHT,BLUE_MID]
x_b=np.arange(len(tags))
bars=ax_b.bar(x_b,r2s,color=bar_c,edgecolor=WHITE,linewidth=0.6,width=0.55)
bars[3].set_edgecolor(BLUE_MID); bars[3].set_linewidth(0.9)
for xi,yi in zip(x_b,r2s):
    ax_b.text(xi,yi+0.010,f"{yi:.3f}",ha="center",va="bottom",fontsize=9,color=BLUE_DARK,fontweight="bold")
ax_b.annotate("",xy=(2.78,no_ln_r2+0.005),xytext=(3.22,full_r2+0.005),
              arrowprops=dict(arrowstyle="<->",color=RED,lw=1.2,shrinkA=2,shrinkB=2))
ax_b.text(3.0,max(no_ln_r2,full_r2)+0.065,f"+{fixed_gain:.3f}",ha="center",va="center",fontsize=9,color=RED,fontweight="bold")
ax_b.set_xticks(x_b); ax_b.set_xticklabels(tags,fontsize=9)
ax_b.set_ylabel("R²",fontsize=11)
ax_b.set_ylim(0,0.6)
ax_b.set_yticks([0,0.15,0.3,0.45,0.6])
plabel(ax_b,"b"); style(ax_b)

# ============ Panel c: 3x4 heatmap ============
temp_order=["<10C","10-20C","20-30C",">=30C"]
soc_order=["<50%","50-70%","70-90%",">=90%"]
temp=cond[cond["dimension"]=="Temperature"].set_index("bin").loc[temp_order]
soc =cond[cond["dimension"]=="SOC"].set_index("bin").loc[soc_order]

# Current quartile deltas (4 bins, n~2400 each)
cur=oof["w30d_abs_current_mean"]; cur_q=pd.qcut(cur,q=4,labels=["Q1","Q2","Q3","Q4"])
cur_d=[]; cur_l=[]
for q in ["Q1","Q2","Q3","Q4"]:
    s=oof[cur_q==q]
    cur_d.append(s["abs_err_no_loadnorm"].mean()-s["abs_err_full"].mean())
    lo,hi=s["w30d_abs_current_mean"].min(),s["w30d_abs_current_mean"].max()
    cur_l.append(f"{lo:.0f}-{hi:.0f}A")

mat=np.full((3,4),np.nan)
mat[0,:]=temp["delta_mae_no_ln_minus_full"].to_numpy()
mat[1,:]=cur_d
mat[2,:]=soc["delta_mae_no_ln_minus_full"].to_numpy()

small_n=np.full((3,4),False)
small_n[0,:]=temp["small_n_flag"].to_numpy()
small_n[2,:]=soc["small_n_flag"].to_numpy()

cell_labels=[
    ["<10C","10-20C","20-30C",">=30C"],
    cur_l,
    ["<50%","50-70%","70-90%",">=90%"],
]

cmap=mpl.colors.LinearSegmentedColormap.from_list("fig4_blue",
    [BLUE_GHOST,BLUE_LIGHT,BLUE_BRIGHT,BLUE_MID,BLUE_DARK])
vmin,vmax=0.01,0.14
im=ax_c.imshow(mat,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto")

for r in range(3):
    for c in range(4):
        v=mat[r,c]
        if np.isnan(v): continue
        txt=f"({v:.3f})" if small_n[r,c] else f"{v:.3f}"
        # Uniform text: always white, bold, consistent fontsize
        ax_c.text(c,r-0.12,txt,ha="center",va="center",
                  fontsize=9,color=WHITE,fontweight="bold")
        ax_c.text(c,r+0.22,cell_labels[r][c],ha="center",va="center",
                  fontsize=7,color=WHITE,fontweight="normal")

ax_c.set_yticks([0,1,2]); ax_c.set_yticklabels(["Temp.","Current","SOC"],fontsize=9)
ax_c.set_xticks(range(4)); ax_c.set_xticklabels(["Low","Moderate","Elevated","High"],fontsize=8)
ax_c.set_xlabel("")

cbar=fig.colorbar(im,ax=ax_c,fraction=0.046,pad=0.04)
cbar.set_label("MAE gain (%)", fontsize=8)
cbar.ax.tick_params(labelsize=8,length=2.5,width=0.7); cbar.outline.set_linewidth(0.7)
plabel(ax_c,"c"); style(ax_c)

for ax in [ax_a,ax_b,ax_c]: ax.tick_params(pad=2)

export.save(fig,"Fig4_loadnorm_mechanism_v30",outdir=str(OUT),formats=("pdf","png"))

(OUT/"Fig4_v30_manifest.json").write_text(json.dumps({
    "figure":"Fig4_loadnorm_mechanism_v30",
    "layout":"2-row: a KDE full top, b ablation bar bot-left, c 3x4 heatmap bot-right",
    "panel_c_current":"Re-binned to quartiles (4 bins, n~2400 each) for uniform 4-col layout",
    "panel_a":"KDE |r_SOH| - LoadNorm features capture SOH signal; dotted lines mark distribution medians; MW p=2.8e-15",
    "panel_b":"Ablation bar - R2 ladder, LoadNorm +0.039",
    "panel_c":"Condition-controlled heatmap - Temp(4)+Current(q4)+SOC(4), all MAE gain positive; values are absolute SOH percentage-point gains shown as %",
    "palette":"Fig2/Fig3 unified blue"
},indent=2),encoding="utf-8")
