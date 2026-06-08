import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


FEATURE_CSV = Path("/root/autodl-tmp/results1/le20_conf_ge_0p55_features_loadnorm_20260502_2324/le20_conf_ge_0p55_loadnorm.csv")
OOF_CSV = Path("/root/autodl-tmp/results1_derived/audit_breakthrough_v2_20260511/breakthrough_audit_v2_lr022_i1800_oof.csv")
RANDOM_FOLDS_CSV = Path("/root/autodl-tmp/results1_derived/audit_breakthrough_v2_20260511/breakthrough_audit_v2_random_3x5_folds.csv")
PARQUET_ROOT = Path("/root/autodl-tmp/cleaned_data_parquet_v3")
OUT_DIR = Path("/root/autodl-tmp/results1/fig1_data_insets_20260529_grey")

NAVY = "#2F3D45"
BLUE = "#586A73"
SKY = "#D8E0E4"
RED = "#6F7C83"
GRAY = "#7B878D"
LIGHT = "#F1F3F4"
INK = "#263238"


def setup():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for font_path in [Path("/root/autodl-tmp/results1/arial.ttf"), Path("/root/autodl-tmp/results1/arialbd.ttf")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axes(ax):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    for txt in list(ax.texts):
        txt.remove()
    ax.tick_params(length=0, labelbottom=False, labelleft=False)


def plot_soc_distribution(features):
    col = "w30d_soc_mean" if "w30d_soc_mean" in features.columns else "soc_s"
    x = pd.to_numeric(features[col], errors="coerce").dropna().clip(0, 100)
    fig, ax = plt.subplots(figsize=(2.15, 1.15))
    ax.hist(x, bins=np.linspace(0, 100, 31), density=True, color=SKY, edgecolor="white", linewidth=0.3)
    median = float(x.median())
    ax.axvline(median, color=RED, ls="--", lw=1.0)
    ax.text(median + 2, ax.get_ylim()[1] * 0.82, f"median {median:.0f}%", color=RED, fontsize=6)
    ax.set_xlabel("SOC (%)", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.set_xlim(0, 100)
    clean_axes(ax)
    save(fig, "soc_distribution_w30d")
    return {"source_column": col, "n": int(len(x)), "median": median}


def find_charge_segment(features, max_tries=120):
    candidates = features[["vin", "t_s"]].dropna().sample(min(max_tries, len(features)), random_state=7)
    for _, row in candidates.iterrows():
        vin, t_s = str(row["vin"]), float(row["t_s"])
        path = PARQUET_ROOT / vin / "part-00000.parquet"
        if not path.exists():
            continue
        cols = ["terminaltime", "chargestatus", "totalcurrent", "maxvoltagebattery", "soc"]
        df = pd.read_parquet(path, columns=cols)
        seg = df[(df["terminaltime"] >= t_s) & (df["terminaltime"] <= t_s + 7200) & (df["chargestatus"] == 1)].copy()
        seg = seg.dropna(subset=["terminaltime", "totalcurrent"])
        if len(seg) >= 80:
            return vin, t_s, seg.iloc[: min(len(seg), 900)].copy()
    raise RuntimeError("No suitable charge segment found")


def plot_charge_curve(features):
    vin, t_s, seg = find_charge_segment(features)
    elapsed = (seg["terminaltime"].to_numpy() - seg["terminaltime"].iloc[0]) / 3600.0
    current = np.abs(seg["totalcurrent"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(1.75, 1.30))
    ax.plot(elapsed, current, color=BLUE, lw=1.2)
    ax.fill_between(elapsed, current, color=SKY, alpha=0.35)
    ax.set_xlabel("t (h)", fontsize=7)
    ax.set_ylabel("|I| (A)", fontsize=7)
    ax.set_xlim(max(0, elapsed.min()), max(elapsed.max(), 0.1))
    ax.text(0.05, 0.85, r"$Q=\int |I|dt$", transform=ax.transAxes, fontsize=8, color=NAVY)
    clean_axes(ax)
    save(fig, "charge_current_segment")
    return {"vin": vin, "t_s": t_s, "rows": int(len(seg)), "duration_h": float(elapsed.max())}


def plot_charge_cumulative_q(features):
    vin, t_s, seg = find_charge_segment(features)
    time_s = seg["terminaltime"].to_numpy(dtype=float)
    elapsed = (time_s - time_s[0]) / 3600.0
    current = np.abs(seg["totalcurrent"].to_numpy(dtype=float))
    dt_h = np.diff(time_s, prepend=time_s[0]) / 3600.0
    dt_h = np.clip(dt_h, 0, 5 / 3600.0)
    q_ah = np.cumsum(current * dt_h)

    fig, ax = plt.subplots(figsize=(1.75, 1.30))
    ax.plot(elapsed, q_ah, color=BLUE, lw=1.25)
    ax.fill_between(elapsed, q_ah, color=SKY, alpha=0.30)
    ax.set_xlabel("t (h)", fontsize=7)
    ax.set_ylabel("Q (Ah)", fontsize=7)
    ax.set_xlim(max(0, elapsed.min()), max(elapsed.max(), 0.1))
    ax.text(0.08, 0.82, r"$Q=\int |I|dt$", transform=ax.transAxes, fontsize=8, color=NAVY)
    clean_axes(ax)
    save(fig, "charge_cumulative_q_segment")
    return {
        "vin": vin,
        "t_s": t_s,
        "rows": int(len(seg)),
        "duration_h": float(elapsed.max()),
        "q_ah_end": float(q_ah[-1]),
    }


def find_discharge_window(features, max_tries=180):
    candidates = features[["vin", "t_s"]].dropna().sample(min(max_tries, len(features)), random_state=11)
    best = None
    best_score = -np.inf
    for _, row in candidates.iterrows():
        vin, t_s = str(row["vin"]), float(row["t_s"])
        path = PARQUET_ROOT / vin / "part-00000.parquet"
        if not path.exists():
            continue
        cols = ["terminaltime", "chargestatus", "totalcurrent", "maxvoltagebattery", "minvoltagebattery", "soc"]
        df = pd.read_parquet(path, columns=cols)
        win = df[(df["terminaltime"] < t_s) & (df["terminaltime"] >= t_s - 10 * 86400) & (df["chargestatus"] == 3)].copy()
        win = win.dropna(subset=["terminaltime", "totalcurrent", "maxvoltagebattery", "minvoltagebattery"])
        if len(win) < 600:
            continue
        win = win.iloc[-1200:].copy()
        spread = (win["maxvoltagebattery"] - win["minvoltagebattery"]).to_numpy(dtype=float)
        abs_i = np.abs(win["totalcurrent"].to_numpy(dtype=float))
        mask = np.isfinite(spread) & np.isfinite(abs_i) & (abs_i > 5)
        if mask.sum() < 300:
            continue
        score = np.nanstd(abs_i[mask]) * np.nanstd(spread[mask])
        if score > best_score:
            best_score = score
            best = (vin, t_s, win.loc[mask].iloc[-500:].copy())
    if best is None:
        raise RuntimeError("No suitable discharge window found")
    return best


def plot_discharge_window_timeline(features):
    vin, t_s, win = find_discharge_window(features)
    time_s = win["terminaltime"].to_numpy(dtype=float)
    rel_days = (time_s - float(t_s)) / 86400.0
    order = np.argsort(rel_days)
    rel_days = rel_days[order]
    if len(rel_days) < 20:
        raise RuntimeError("Not enough discharge records for window timeline")

    gaps = np.where(np.diff(rel_days) > (10.0 / 1440.0))[0]
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps, len(rel_days) - 1]
    segments = [(float(rel_days[s]), float(rel_days[e])) for s, e in zip(starts, ends) if e > s]

    fig, ax = plt.subplots(figsize=(2.55, 0.95), dpi=450)
    for start, end in segments:
        ax.barh(0, end - start, left=start, height=0.36, color=BLUE, alpha=0.82)
    ax.axvline(0, color=RED, lw=0.9, ls="--")
    ax.set_xlim(-10, 0.25)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([])
    ax.set_xlabel("Days before label", fontsize=7)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0, labelbottom=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    save(fig, "discharge_window_timeline")
    return {
        "vin": vin,
        "t_s": float(t_s),
        "n_records": int(len(win)),
        "n_segments": int(len(segments)),
        "window_days": 10,
    }


def plot_raw_vs_loadnorm(features):
    vin, t_s, win = find_discharge_window(features)
    t = (win["terminaltime"].to_numpy() - win["terminaltime"].iloc[0]) / 60.0
    spread = (win["maxvoltagebattery"] - win["minvoltagebattery"]).to_numpy(dtype=float) * 1000.0
    abs_i = np.abs(win["totalcurrent"].to_numpy(dtype=float))
    loadnorm = spread / np.maximum(abs_i, 1e-6)
    keep = np.linspace(0, len(t) - 1, min(220, len(t))).astype(int)
    t, spread, loadnorm = t[keep], spread[keep], loadnorm[keep]

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.20), sharex=False)
    axes[0].plot(t, spread, color=BLUE, lw=1.0)
    axes[0].set_title(r"Raw spread ($\sigma_V$)", fontsize=7)
    axes[0].set_ylabel("mV", fontsize=7)
    axes[0].set_xlabel("Time (min)", fontsize=7)
    axes[1].plot(t, loadnorm, color=NAVY, lw=1.0)
    axes[1].set_title(r"Load-normalized ($\sigma_V/|I|$)", fontsize=7)
    axes[1].set_ylabel("mV/A", fontsize=7)
    axes[1].set_xlabel("Time (min)", fontsize=7)
    for ax in axes:
        clean_axes(ax)
    fig.tight_layout(w_pad=1.0)
    save(fig, "raw_vs_loadnorm_discharge_example")

    fig_raw, ax_raw = plt.subplots(figsize=(1.55, 1.20))
    ax_raw.plot(t, spread, color=BLUE, lw=1.0)
    ax_raw.set_xlabel("Time (min)", fontsize=7)
    ax_raw.set_ylabel(r"$\sigma_V$ (mV)", fontsize=7)
    clean_axes(ax_raw)
    save(fig_raw, "raw_voltage_spread_curve")

    fig_ln, ax_ln = plt.subplots(figsize=(1.55, 1.20))
    ax_ln.plot(t, loadnorm, color=NAVY, lw=1.0)
    ax_ln.set_xlabel("Time (min)", fontsize=7)
    ax_ln.set_ylabel(r"$\sigma_V/|I|$ (mV/A)", fontsize=7)
    clean_axes(ax_ln)
    save(fig_ln, "loadnorm_spread_curve")

    return {
        "vin": vin,
        "t_s": t_s,
        "rows": int(len(win)),
        "spread_mV_median": float(np.nanmedian(spread)),
        "loadnorm_mV_per_A_median": float(np.nanmedian(loadnorm)),
        "split_outputs": ["raw_voltage_spread_curve", "loadnorm_spread_curve"],
    }


def oof_columns(oof):
    true_col = "y_true" if "y_true" in oof.columns else "soh"
    pred_col = "y_pred" if "y_pred" in oof.columns else "pred"
    if pred_col not in oof.columns:
        pred_candidates = [c for c in oof.columns if "pred" in c.lower()]
        if not pred_candidates:
            raise RuntimeError(f"No prediction column in {oof.columns.tolist()}")
        pred_col = pred_candidates[0]
    return true_col, pred_col


def plot_scatter_and_cdf(oof):
    true_col, pred_col = oof_columns(oof)
    y = pd.to_numeric(oof[true_col], errors="coerce")
    p = pd.to_numeric(oof[pred_col], errors="coerce")
    mask = y.notna() & p.notna()
    y, p = y[mask].to_numpy(), p[mask].to_numpy()
    err = np.abs(p - y)

    fig, ax = plt.subplots(figsize=(1.55, 1.35))
    ax.scatter(y, p, s=2, color=BLUE, alpha=0.35, rasterized=True)
    lo, hi = 72, 102
    ax.plot([lo, hi], [lo, hi], color=RED, lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True SOHproxy (%)", fontsize=6.5)
    ax.set_ylabel("Predicted SOHproxy (%)", fontsize=6.5)
    ax.text(0.04, 0.92, r"$R^2=0.501$", transform=ax.transAxes, fontsize=7, color=NAVY)
    clean_axes(ax)
    save(fig, "true_vs_pred_lr022_i1800")

    fig, ax = plt.subplots(figsize=(1.35, 1.35))
    x = np.sort(err)
    yy = np.linspace(0, 1, len(x))
    ax.plot(x, yy, color=NAVY, lw=1.2)
    p80, p90 = np.percentile(err, [80, 90])
    for val, lab in [(p80, "P80"), (p90, "P90")]:
        ax.axvline(val, color=GRAY, lw=0.7, ls=":")
        ax.text(val, 0.07, lab, rotation=90, fontsize=5.5, color=GRAY, va="bottom")
    ax.set_xlabel("|Error| (% SOH)", fontsize=6.5)
    ax.set_ylabel("CDF", fontsize=6.5)
    ax.set_xlim(0, max(6, np.percentile(err, 98)))
    ax.set_ylim(0, 1.01)
    clean_axes(ax)
    save(fig, "absolute_error_cdf_lr022_i1800")
    return {"n": int(mask.sum()), "mae": float(np.mean(err)), "p80_abs_error": float(p80), "p90_abs_error": float(p90)}


def plot_fold_r2():
    folds = pd.read_csv(RANDOM_FOLDS_CSV)
    if "r2" not in folds.columns:
        raise RuntimeError("random folds CSV missing r2")
    r2 = pd.to_numeric(folds["r2"], errors="coerce").dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(2.10, 1.25))
    x = np.arange(1, len(r2) + 1)
    ax.scatter(x, r2, s=14, color=BLUE, zorder=3)
    ax.plot(x, r2, color=SKY, lw=0.9, zorder=2)
    mean = float(np.mean(r2))
    ax.axhline(mean, color=NAVY, lw=1.0, ls="--")
    ax.fill_between([0.5, len(r2) + 0.5], mean - np.std(r2), mean + np.std(r2), color=SKY, alpha=0.25, lw=0)
    ax.set_xlabel("Fold ID", fontsize=7)
    ax.set_ylabel(r"$R^2$", fontsize=7)
    ax.set_xlim(0.5, len(r2) + 0.5)
    ax.set_ylim(min(0.45, r2.min() - 0.01), max(0.515, r2.max() + 0.01))
    ax.text(0.05, 0.88, f"mean={mean:.3f}", transform=ax.transAxes, fontsize=6.5, color=NAVY)
    clean_axes(ax)
    save(fig, "random_vin_3x5_fold_r2")
    return {"n_folds": int(len(r2)), "mean_r2": mean, "std_r2": float(np.std(r2)), "min_r2": float(np.min(r2)), "max_r2": float(np.max(r2))}


def main():
    setup()
    features = pd.read_csv(FEATURE_CSV)
    oof = pd.read_csv(OOF_CSV)
    manifest = {
        "feature_csv": str(FEATURE_CSV),
        "oof_csv": str(OOF_CSV),
        "random_folds_csv": str(RANDOM_FOLDS_CSV),
        "parquet_root": str(PARQUET_ROOT),
        "outputs": {},
    }
    manifest["outputs"]["soc_distribution_w30d"] = plot_soc_distribution(features)
    manifest["outputs"]["charge_current_segment"] = plot_charge_curve(features)
    manifest["outputs"]["charge_cumulative_q_segment"] = plot_charge_cumulative_q(features)
    manifest["outputs"]["discharge_window_timeline"] = plot_discharge_window_timeline(features)
    manifest["outputs"]["raw_vs_loadnorm_discharge_example"] = plot_raw_vs_loadnorm(features)
    manifest["outputs"]["true_vs_pred_and_cdf"] = plot_scatter_and_cdf(oof)
    manifest["outputs"]["random_vin_3x5_fold_r2"] = plot_fold_r2()
    (OUT_DIR / "fig1_data_insets_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
