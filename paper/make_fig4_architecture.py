"""Fig. 4 of the revised manuscript: HyPhysML architecture (revision 3)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(8.6, 10.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 120); ax.axis("off")


def box(x, y, w, h, title, body, fc, ec="#333333", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                                fc=fc, ec=ec, lw=1.3, ls=ls))
    if body:
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontweight="bold", fontsize=9.5)
        ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=8.2, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontweight="bold", fontsize=9.5)


def arrow(x1, y1, x2, y2, ls="-", c="#333333"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, ls=ls, shrinkA=2, shrinkB=2))


box(5, 106, 90, 11, "Input: 6,656 laboratory records (training split of each seed)",
    "Raw design variables: insulator type (CD, AD, CF), RH, aging, J, K, SDD   →   target FOV (kV)", "#EEF3FA")
arrow(30, 106, 30, 101); arrow(77.5, 106, 77.5, 101)
box(5, 88, 50, 13, "Feature engineering (25 features)",
    "logs, products, ratios, composite env_stress\n(Obenaus-motivated; Eqs. 1–2) + type indicators", "#EEF3FA")
box(60, 88, 35, 13, "Raw inputs only (12)", "8 numerical variables +\n4 type indicators", "#FDEDEC")
arrow(30, 88, 30, 83)
box(5, 70, 50, 13, "Layer 1 — nested Bayesian HPO",
    "Optuna-TPE, 200 trials per learner,\n5-fold CV inside each training set only\n(test set never used)", "#E8F5E9")
arrow(30, 70, 30, 65)
box(5, 41, 50, 24, "Layer 2 — out-of-fold stacking",
    "XGBoost · LightGBM · GBR · HistGBR\nRF · Extra Trees · KNN\n"
    "↓  5-fold OOF prediction matrix M\nridge meta-learner (α = 0.1)\n"
    "ŷ = w₀ + Σ w_b f_b(x)", "#E8F5E9")
arrow(30, 41, 30, 34)
box(5, 27, 50, 6, "HyPhysML prediction  ŷ (kV)", "", "#FFF8E1")

arrow(77.5, 88, 77.5, 65)
box(60, 41, 35, 24, "HyPhysML-MC (constrained)",
    "XGBoost · LightGBM · HistGBR\nmonotone constraints:\nFOV non-increasing in\n"
    "SDD, RH, aging, J\nnon-negative ridge weights", "#FDEDEC")
arrow(77.5, 41, 77.5, 34)
box(60, 27, 35, 6, "Monotone prediction  ŷ_MC", "", "#FFF8E1")

# tuned hyperparameters reused by the constrained variant
ax.annotate("", xy=(60, 58), xytext=(55, 74),
            arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.2, ls="--"))
ax.text(56.3, 68.5, "tuned\nsettings", fontsize=7.2, color="#2E7D32")

box(5, 1, 90, 18, "Layer 3 — post-hoc physical checks (do NOT alter the predictions)",
    "(a) auxiliary log-linear Obenaus regression: signs of β for ln SDD, ln RH, aging, ln J, ln CD, ln AD (Eq. 6)\n"
    "(b) prediction-level monotonicity audit: share of level steps in SDD, RH, aging and J\n"
    "where the predicted FOV rises\n"
    "(c) raw-input response curves and leave-one-level-out extrapolation tests",
    "#F3F3F3", ls="--")
arrow(30, 27, 30, 19.5, ls="--", c="#666666")
arrow(77.5, 27, 77.5, 19.5, ls="--", c="#666666")

plt.savefig("revision3_latex/figs/Fig4_architecture.png", dpi=300, bbox_inches="tight")
print("ok")
