# -*- coding: utf-8 -*-
"""
Figures S1-S6 of the supplementary material and Figs. R3.C1.1-R3.C6.1 of the
response letter (TII-26-8813).

The plotted values are stored in DATA below. To redraw all figures, run

    python make_figures.py

Each figure is written to ./response_assets/ in four formats:
    *.pdf  vector, used by the LaTeX response and supplementary material
    *.png  600 dpi raster
    *.svg  vector, text converted to outlines
    *.emf  vector (Windows Enhanced Metafile), converted from the SVG with
           Inkscape (>= 1.0). If Inkscape is not on the PATH, the EMF step is
           skipped and a message is printed.
"""
import os
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "response_assets")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# DATA (MAE values are x10^2, averaged over 30 runs)
# ----------------------------------------------------------------------------
LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 8.0, 10.0]
DATA = {
    # Comment 2: composition of the injected degradation, dataset average.
    # missing / displaced: % of grid positions; type shares: % within each part.
    "comp": {
        "level": LEVELS,
        "missing": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 2.0, 4.0, 5.5],
        "displaced": [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 3.0, 4.0, 4.5],
        "miss_isolated": [68, 64, 59, 55, 50, 45, 38, 30, 25],
        "miss_burst": [22, 24, 26, 27, 29, 30, 32, 32, 31],
        "miss_long": [10, 12, 15, 18, 21, 25, 30, 38, 44],
        "disp_single": [73, 69, 66, 63, 58, 53, 48, 42, 38],
        "disp_accum": [20, 22, 24, 26, 29, 32, 35, 38, 40],
        "disp_reset": [7, 9, 10, 11, 13, 15, 17, 20, 22],
        "isolated_share": [71, 67, 63, 60, 55, 50, 44, 36, 31],
    },
    # Comment 4: low affected-data ratios, six-dataset average
    "low": {
        "ratio": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "COOL": [1.12, 1.24, 1.36, 1.48, 1.61, 1.77],
        "DSTGNN": [1.30, 1.41, 1.52, 1.63, 1.76, 1.90],
        "TiDAL-Net w/o Bi-NSDE": [1.27, 1.39, 1.51, 1.63, 1.76, 1.88],
        "TiDAL-Net": [1.21, 1.31, 1.41, 1.51, 1.60, 1.68],
        "latency_full": [5.8] * 6,
        "latency_gated": [3.6, 3.7, 3.9, 4.0, 4.2, 4.3],
        # MAE difference TiDAL-Net minus COOL at every level (5-10% from Table II)
        "diff_level": LEVELS,
        "diff": [0.09, 0.07, 0.05, 0.03, -0.01, -0.09, -0.27, -0.52, -0.62],
        # MAE by position type, averaged over 0.5%-2.5%
        "types": ["Isolated\npositions", "Consecutive\nruns", "Displaced\nsamples"],
        "type_COOL": [0.92, 2.12, 1.38],
        "type_TiDAL": [1.15, 1.84, 1.12],
    },
    # Comment 5: contiguous gaps, five primary datasets, 5% missing
    "gap": {
        "G": [8, 16, 32, 64, 128],
        "SAITS": [2.49, 2.98, 3.71, 4.72, 5.94],
        "MAD-ODE": [2.31, 2.70, 3.24, 4.05, 5.10],
        "DSTGNN": [2.15, 2.55, 3.14, 3.96, 5.02],
        "COOL": [2.10, 2.47, 3.03, 3.82, 4.85],
        "TiDAL-Net": [1.88, 2.14, 2.52, 3.07, 3.86],
        "NA": {"SAITS": 6.37, "MAD-ODE": 5.82, "DSTGNN": 5.41, "COOL": 5.36, "TiDAL-Net": 4.71},
        "NAN": {"SAITS": 6.58, "MAD-ODE": 6.47, "DSTGNN": 6.41, "COOL": 6.35, "TiDAL-Net": 6.12},
        "coverage": [90.1, 89.6, 88.9, 87.8, 86.4],
    },
    # Comment 3: uncertainty reliability
    "unc": {
        "datasets": ["MSL", "SMAP", "ESA-ADB", "Real-A", "Real-B", "Const-Sim"],
        "rho": [0.71, 0.66, 0.63, 0.68, 0.73, 0.65],
        "rho_bin": [0.52, 0.47, 0.41, 0.49, 0.56, 0.44],
        "cov_raw": [93.8, 92.6, 87.4, 91.9, 94.3, 88.6],
        "cov_scaled": [90.4, 89.8, 89.1, 90.2, 90.6, 89.3],
        "cmp_methods": ["CSDI", "w/o\ndiffusion", "TiDAL-\nNet"],
        "cmp_crps": [2.12, 1.98, 1.56],
    },
    # Comment 1: mission-mode shift, average over Real-A, Real-B, Const-Sim
    "shift": {
        "methods": ["Fixed GAT", "COOL", "DSTGNN", "TiDAL-Net", "Dense\n$\\mathcal{E}^{(0)}$", "Rebuilt\n$\\mathcal{E}^{(0)}$"],
        "seen": [2.30, 2.19, 2.19, 1.89, 2.01, 1.88],
        "unseen": [2.99, 2.70, 2.65, 2.33, 2.21, 2.06],
        "increase": [29.7, 23.4, 21.0, 23.5, 10.3, 9.4],
        "datasets": ["Real-A", "Real-B", "Const-Sim"],
        "inc_involved": [38.2, 35.7, 50.9],
        "inc_other": [13.1, 12.6, 17.2],
    },
    # Comment 6: graph-related parameters (MAE x1, not x10^2)
    "sens": {
        "gamma": {"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "mae": [0.0231, 0.0224, 0.0220, 0.0223, 0.0232, 0.0247],
                  "lat": [7.1, 6.3, 5.8, 5.5, 5.3, 5.1], "default": 0.3, "label": r"$\gamma$", "log": False},
        "kmic": {"x": [4, 6, 8, 12, 16, 24], "mae": [0.0238, 0.0226, 0.0220, 0.0221, 0.0225, 0.0233],
                 "lat": [5.1, 5.4, 5.8, 6.4, 7.0, 8.1], "default": 8, "label": r"$K_{\mathrm{MIC}}$", "log": True},
        "tr": {"x": [2, 5, 10, 20, 50, 100], "mae": [0.0243, 0.0227, 0.0220, 0.0222, 0.0229, 0.0236],
               "lat": None, "default": 10, "label": r"$T_r$", "log": True},
        "gt": {"x": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0], "mae": [0.0230, 0.0223, 0.0220, 0.0222, 0.0229, 0.0241],
               "lat": None, "default": 0.1, "label": r"$\gamma_t$", "log": True},
        "dtmax": {"x": [2, 5, 10, 20, 40, 160], "mae": [0.0232, 0.0217, 0.0210, 0.0213, 0.0221, 0.0236],
                  "lat": None, "default": 10, "label": r"$\Delta t_{\max}$", "log": True},
    },
}

# ----------------------------------------------------------------------------
# Style. Methods keep the identity colors of the manuscript (TiDAL-Net red,
# COOL blue); a third highlighted entity uses yellow, and other baselines are
# neutral grays with distinct markers and line styles. Ordered categories use
# one-hue ramps (blue for missing data, orange for displacement).
# ----------------------------------------------------------------------------
RED, BLUE, YELLOW = "#e34948", "#2a78d6", "#eda100"
RED_DARK = "#a8302f"
GRAY1, GRAY2 = "#52514e", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#9ec5f4", "#3987e5", "#184f95"]
ORANGE_RAMP = ["#f6c1a6", "#eb6834", "#a8441c"]

STYLE = {
    "TiDAL-Net": dict(color=RED, marker="o", ls="-", lw=2.0, zorder=5),
    "COOL": dict(color=BLUE, marker="s", ls="--", lw=1.6, zorder=4),
    "DSTGNN": dict(color=YELLOW, marker="^", ls="-.", lw=1.4, zorder=3),
    "TiDAL-Net w/o Bi-NSDE": dict(color=GRAY1, marker="D", ls=":", lw=1.4, zorder=3),
    "MAD-ODE": dict(color=GRAY1, marker="v", ls=":", lw=1.4, zorder=2),
    "SAITS": dict(color=GRAY2, marker="P", ls=(0, (5, 2)), lw=1.4, zorder=2),
}

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
})

INKSCAPE = shutil.which("inkscape")


def save(fig, name):
    base = os.path.join(OUT, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".svg")
    plt.close(fig)
    if INKSCAPE:
        subprocess.run([INKSCAPE, base + ".svg", "--export-type=emf", "--export-filename=" + base + ".emf"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("Inkscape not found: EMF skipped for", name)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", color=INK)


def line(ax, x, y, name, label=None):
    s = STYLE[name]
    ax.plot(x, y, label=label or name, color=s["color"], marker=s["marker"], ls=s["ls"],
            lw=s["lw"], zorder=s["zorder"], markeredgecolor="white", markeredgewidth=0.6)


def bars(ax, xs, vals, width, color, label, bottom=None):
    # Hatching is not used: Inkscape's EMF export drops the fill under hatch
    # patterns, and the validated colors already separate every pair.
    return ax.bar(xs, vals, width=width, color=color, label=label, edgecolor="white",
                  linewidth=1.0, zorder=3, bottom=bottom)


def level_labels(levels):
    return [f"{v:g}%" for v in levels]


# ---------------------------------------------------------------- Comment 2
def fig_composition():
    d = DATA["comp"]
    x = np.arange(len(d["level"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    w = 0.66
    ax = axes[0]
    miss = np.array(d["missing"])
    parts = [np.array(d["miss_isolated"]), np.array(d["miss_burst"]), np.array(d["miss_long"])]
    names = ["Isolated points", "Bursts of 2–8 samples", "Long gaps"]
    bottom = np.zeros(len(x))
    for part, name, col in zip(parts, names, BLUE_RAMP):
        v = miss * part / 100.0
        bars(ax, x, v, w, col, name, bottom=bottom)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels(d["level"]), rotation=45)
    ax.set_xlabel("Affected-data ratio")
    ax.set_ylabel("Share of grid positions (%)")
    ax.set_ylim(0, 6.6)
    ax.grid(axis="x", visible=False)
    panel_title(ax, "(a) Missing data")
    ax.legend(loc="upper left")

    ax = axes[1]
    disp = np.array(d["displaced"])
    parts = [np.array(d["disp_single"]), np.array(d["disp_accum"]), np.array(d["disp_reset"])]
    names = ["Single offsets (1–3 intervals)", "Accumulated segments (≤4)", "Blocks after resets"]
    bottom = np.zeros(len(x))
    for part, name, col in zip(parts, names, ORANGE_RAMP):
        v = disp * part / 100.0
        bars(ax, x, v, w, col, name, bottom=bottom)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels(d["level"]), rotation=45)
    ax.set_xlabel("Affected-data ratio")
    ax.set_ylabel("Share of grid positions (%)")
    ax.set_ylim(0, 6.6)
    ax.grid(axis="x", visible=False)
    panel_title(ax, "(b) Timestamp displacement")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.plot(x, d["isolated_share"], color=INK2, marker="o", lw=1.8, markeredgecolor="white", markeredgewidth=0.6)
    for xi, v in zip(x, d["isolated_share"]):
        ax.text(xi, v + 2.2, f"{v}", ha="center", va="bottom", fontsize=6.6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels(d["level"]), rotation=45)
    ax.set_xlabel("Affected-data ratio")
    ax.set_ylabel("Isolated share of affected positions (%)")
    ax.set_ylim(20, 82)
    panel_title(ax, "(c) Isolated share")
    fig.tight_layout(w_pad=1.0)
    save(fig, "R3_C2_composition")


# ---------------------------------------------------------------- Comment 4
def fig_low_ratio():
    d = DATA["low"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    ax = axes[0, 0]
    for name in ["TiDAL-Net w/o Bi-NSDE", "DSTGNN", "COOL", "TiDAL-Net"]:
        line(ax, d["ratio"], d[name], name)
    ax.set_xlabel("Affected-data ratio (%)")
    ax.set_ylabel(r"MAE ($\times10^{2}$)")
    ax.set_xticks(d["ratio"])
    panel_title(ax, "(a) Six-dataset average, 0.5%–3.0%")
    ax.legend(loc="upper left", handlelength=2.6)

    ax = axes[0, 1]
    iso = DATA["comp"]["isolated_share"]
    diff = d["diff"]
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(iso, diff, color=RED, marker="o", lw=1.8, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    show = {0.5: (0, 6), 1.5: (0, 6), 2.5: (0, 6), 3.0: (6, 2), 5.0: (6, 2), 8.0: (6, 2), 10.0: (-4, -12)}
    for xi, yi, lv in zip(iso, diff, d["diff_level"]):
        if lv in show:
            ax.annotate(f"{lv:g}%", (xi, yi), textcoords="offset points", xytext=show[lv],
                        ha="center", fontsize=6.6, color=INK2)
    ax.text(62, -0.45, "Below zero: TiDAL-Net\nmore accurate than COOL", fontsize=6.8, color=INK2, ha="center")
    ax.set_xlabel("Isolated share of affected positions (%)")
    ax.set_ylabel(r"MAE difference ($\times10^{2}$)")
    ax.set_xlim(28, 76)
    ax.set_ylim(-0.72, 0.2)
    ax.invert_xaxis()
    panel_title(ax, "(b) TiDAL-Net minus COOL vs. composition")

    ax = axes[1, 0]
    x = np.arange(len(d["types"]))
    w = 0.36
    b1 = bars(ax, x - w / 2, d["type_COOL"], w, BLUE, "COOL")
    b2 = bars(ax, x + w / 2, d["type_TiDAL"], w, RED, "TiDAL-Net")
    for bb in list(b1) + list(b2):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.04, f"{bb.get_height():.2f}",
                ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(d["types"])
    ax.set_ylabel(r"MAE ($\times10^{2}$)")
    ax.set_ylim(0, 3.0)
    ax.grid(axis="x", visible=False)
    panel_title(ax, "(c) By position type, 0.5%–2.5%")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.plot(d["ratio"], d["latency_full"], color=RED, marker="o", lw=2.0, label="Full Bi-NSDE",
            markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(d["ratio"], d["latency_gated"], color=GRAY1, marker="D", ls="--", lw=1.6, label="Gated inference",
            markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel("Affected-data ratio (%)")
    ax.set_ylabel("Single-window latency (ms)")
    ax.set_xticks(d["ratio"])
    ax.set_ylim(0, 7)
    panel_title(ax, "(d) Inference latency")
    ax.legend(loc="lower right")
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    save(fig, "R3_C4_low_ratio")


# ---------------------------------------------------------------- Comment 5
def fig_long_gap():
    d = DATA["gap"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.3, 1.1, 0.9]})
    ax = axes[0]
    for name in ["SAITS", "MAD-ODE", "DSTGNN", "COOL", "TiDAL-Net"]:
        line(ax, d["G"], d[name], name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(d["G"])
    ax.set_xticklabels([str(g) for g in d["G"]])
    ax.minorticks_off()
    ax.set_xlabel(r"Gap length $G$ (nominal intervals)")
    ax.set_ylabel(r"MAE ($\times10^{2}$)")
    panel_title(ax, "(a) Contiguous gaps")
    ax.legend(loc="upper left", handlelength=2.6)

    ax = axes[1]
    names = ["SAITS", "MAD-ODE", "DSTGNN", "COOL", "TiDAL-Net"]
    x = np.arange(len(names))
    w = 0.38
    na = [d["NA"][n] for n in names]
    nan = [d["NAN"][n] for n in names]
    bars(ax, x - w / 2, na, w, GRAY2, "No anchor")
    bars(ax, x + w / 2, nan, w, GRAY1, "No anchor, neighbors lost")
    ax.bar(x[-1] - w / 2, na[-1], w, color=RED, edgecolor="white", linewidth=1.0, zorder=4)
    ax.bar(x[-1] + w / 2, nan[-1], w, color=RED_DARK, edgecolor="white", linewidth=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["SAITS", "MAD-\nODE", "DST-\nGNN", "COOL", "TiDAL-\nNet"], fontsize=7)
    ax.set_ylabel(r"MAE ($\times10^{2}$)")
    ax.set_ylim(0, 9.6)
    ax.grid(axis="x", visible=False)
    panel_title(ax, "(b) Without valid anchors")
    ax.legend(loc="upper left", ncol=1)

    ax = axes[2]
    ax.plot(d["G"], d["coverage"], color=RED, marker="o", lw=2.0, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(90, color=INK2, lw=0.8, ls="--")
    ax.text(8, 90.25, "nominal 90%", fontsize=6.8, color=INK2, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(d["G"])
    ax.set_xticklabels([str(g) for g in d["G"]])
    ax.minorticks_off()
    ax.set_ylim(84, 92)
    ax.set_xlabel(r"Gap length $G$")
    ax.set_ylabel("90% interval coverage (%)")
    panel_title(ax, "(c) Coverage in gaps")
    fig.tight_layout(w_pad=1.2)
    save(fig, "R3_C5_long_gap")


# ---------------------------------------------------------------- Comment 3
def fig_uncertainty():
    d = DATA["unc"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.25, 1.25, 0.8]})
    x = np.arange(len(d["datasets"]))
    w = 0.38
    ax = axes[0]
    bars(ax, x - w / 2, d["cov_raw"], w, GRAY2, "Raw")
    bars(ax, x + w / 2, d["cov_scaled"], w, RED, "Scaled")
    ax.axhline(90, color=INK, lw=0.8, ls="--")
    ax.set_ylim(80, 99)
    ax.set_xticks(x)
    ax.set_xticklabels(d["datasets"], rotation=30, ha="right")
    ax.set_ylabel("90% interval coverage (%)")
    ax.grid(axis="x", visible=False)
    panel_title(ax, "(a) Interval coverage")
    ax.legend(loc="upper left", ncol=2)

    ax = axes[1]
    bars(ax, x - w / 2, d["rho"], w, RED, "All positions")
    bars(ax, x + w / 2, d["rho_bin"], w, GRAY1, "Equal anchor distance")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(d["datasets"], rotation=30, ha="right")
    ax.set_ylabel(r"Spearman $\rho(\sigma_{i,k},|e_{i,k}|)$")
    ax.grid(axis="x", visible=False)
    panel_title(ax, "(b) Rank correlation")
    ax.legend(loc="upper left", ncol=1)

    ax = axes[2]
    xm = np.arange(len(d["cmp_methods"]))
    bb = ax.bar(xm, d["cmp_crps"], width=0.6, color=[GRAY2, GRAY1, RED], edgecolor="white", linewidth=1.0, zorder=3)
    for b in bb:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.set_xticks(xm)
    ax.set_xticklabels(d["cmp_methods"], fontsize=6.8)
    ax.set_ylabel(r"CRPS ($\times10^{2}$)")
    ax.set_ylim(0, 2.6)
    ax.grid(axis="x", visible=False)
    panel_title(ax, "(c) CRPS")
    fig.tight_layout(w_pad=1.2)
    save(fig, "R3_C3_uncertainty")


# ---------------------------------------------------------------- Comment 1
def fig_mode_shift():
    d = DATA["shift"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.6, 1.0]})
    ax = axes[0]
    x = np.arange(len(d["methods"]))
    w = 0.38
    bars(ax, x - w / 2, d["seen"], w, BLUE, "Seen modes")
    bars(ax, x + w / 2, d["unseen"], w, YELLOW, "Unseen modes")
    for i, (u, inc) in enumerate(zip(d["unseen"], d["increase"])):
        ax.text(i + w / 2, u + 0.04, f"+{inc:.1f}%", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(d["methods"], fontsize=7)
    ax.set_ylabel(r"MAE ($\times10^{2}$)")
    ax.set_ylim(0, 4.0)
    ax.grid(axis="x", visible=False)
    panel_title(ax, "(a) Seen and unseen mission modes")
    ax.legend(loc="upper right", ncol=2)

    ax = axes[1]
    x = np.arange(len(d["datasets"]))
    bars(ax, x - w / 2, d["inc_involved"], w, RED, "Channels in absent relations")
    bars(ax, x + w / 2, d["inc_other"], w, GRAY2, "Other channels")
    ax.set_xticks(x)
    ax.set_xticklabels(d["datasets"])
    ax.set_ylabel("MAE increase of TiDAL-Net (%)")
    ax.set_ylim(0, 68)
    ax.grid(axis="x", visible=False)
    panel_title(ax, "(b) Channel-level increase")
    ax.legend(loc="upper left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "R3_C1_mode_shift")


# ---------------------------------------------------------------- Comment 6
def fig_sensitivity():
    d = DATA["sens"]
    keys = ["gamma", "kmic", "tr", "gt", "dtmax"]
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.35, 1.0], hspace=0.55, wspace=0.55)
    for j, k in enumerate(keys):
        p = d[k]
        ax = fig.add_subplot(gs[0, j])
        ax.plot(p["x"], np.array(p["mae"]) * 100, color=RED, marker="o", lw=2.0,
                markeredgecolor="white", markeredgewidth=0.6)
        i0 = p["x"].index(p["default"])
        ax.plot([p["x"][i0]], [p["mae"][i0] * 100], marker="*", ms=10, color=INK, zorder=6)
        if p["log"]:
            ax.set_xscale("log")
        ax.set_xticks(p["x"])
        if k == "dtmax":
            ax.set_xticklabels(["2", "5", "10", "20", "40", r"$\infty$"])
        else:
            ax.set_xticklabels([f"{v:g}" for v in p["x"]])
        ax.minorticks_off()
        ax.tick_params(axis="x", labelsize=6.3, rotation=45)
        ax.set_ylim(2.0, 2.55)
        if j == 0:
            ax.set_ylabel(r"MAE ($\times10^{2}$)")
        ax.set_title(f"({chr(97 + j)}) " + p["label"], fontsize=8, loc="left", fontweight="bold")
        if k == "dtmax":
            ax.text(0.03, 0.95, "Const-Sim", transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
        ax2 = fig.add_subplot(gs[1, j])
        if p["lat"] is not None:
            ax2.plot(p["x"], p["lat"], color=GRAY1, marker="D", lw=1.6, ls="--",
                     markeredgecolor="white", markeredgewidth=0.6)
            if p["log"]:
                ax2.set_xscale("log")
            ax2.set_xticks(p["x"])
            ax2.set_xticklabels([f"{v:g}" for v in p["x"]])
            ax2.minorticks_off()
            ax2.tick_params(axis="x", labelsize=6.3, rotation=45)
            ax2.set_ylim(4.5, 8.6)
            if j == 0:
                ax2.set_ylabel("Latency (ms)")
        else:
            ax2.axis("off")
            ax2.text(0.5, 0.55, "Latency unchanged\n(5.8 ms)", ha="center", va="center",
                     fontsize=7, color=INK2, transform=ax2.transAxes)
    save(fig, "R3_C6_sensitivity")


if __name__ == "__main__":
    fig_composition()
    fig_low_ratio()
    fig_long_gap()
    fig_uncertainty()
    fig_mode_shift()
    fig_sensitivity()
    print("written to", OUT)
