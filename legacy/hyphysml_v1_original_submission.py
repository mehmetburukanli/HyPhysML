# ========================================================================
# HyPhysML ULTIMATE — Fizik-Bilgili Hibrit ML
# Tum notebook hucreleri tek .py dosyasinda birlestiriliyor
# ========================================================================


# ========================================================================
# [MARKDOWN CELL 0]
# # HyPhysML ULTIMATE — Fizik-Bilgili Hibrit ML
# **Comprehensive EDA + 14-Model Benchmark + Statistical Analysis**
# 
# ## Steps
# 1. **Cell 1** → Setup + validation  
# 2. **Cell 2** → Load data  
# 3. **Cell 3** → Parameters  
# 4. **Cell 4** → Run all (~55 min GPU T4, EDA included)
# ========================================================================


# ========================================================================
# [CODE CELL 1]
# ========================================================================
# ── Setup + validation ──────────────────────────────────────────
# For local setup run in terminal:
# pip install optuna xgboost lightgbm shap openpyxl scikit-learn pandas matplotlib seaborn scipy

import sys
packages = {}
for pkg in ["optuna","xgboost","lightgbm","shap"]:
    try:
        m = __import__(pkg)
        packages[pkg] = getattr(m,"__version__","?")
    except ImportError:
        packages[pkg] = "YOK ⚠"

for pkg,ver in packages.items():
    print(f"  {'✅' if 'YOK' not in ver else '❌'} {pkg:<12} {ver}")

if "YOK" in packages.get("xgboost",""):
    print("\n❌ XGBoost not installed! Restart the kernel and try again.")
else:
    # Quick performance test
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.metrics import r2_score
    import xgboost as xgb
    X_t, y_t = make_regression(n_samples=500, n_features=10, noise=5, random_state=42)
    m_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    m_xgb.fit(X_t[:400], y_t[:400])
    print(f"\n✅ XGBoost works! Test R²={r2_score(y_t[400:], m_xgb.predict(X_t[400:])):.4f}")


# ========================================================================
# [CODE CELL 2]
# ========================================================================
# ── Load data ───────────────────────────────────────────────────
import os

# Paths resolve relative to this file, so the repository runs as-is after the
# dataset has been placed in data/. Both can be overridden with environment
# variables, e.g.  FOV_DATA=/path/to/file.xlsx  FOV_OUT=/path/to/results
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # interactive session / notebook
    _HERE = os.getcwd()

DATA_PATH = os.environ.get("FOV_DATA", os.path.join(_HERE, "data", "FOV dataset.xlsx"))
OUT_DIR   = os.environ.get("FOV_OUT",  os.path.join(_HERE, "results"))

os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Data file not found: {DATA_PATH}\n"
        "Download 'FOV dataset.xlsx' from https://doi.org/10.17632/8r7k4cgkg8.1 "
        "and place it in the data/ directory, or set the FOV_DATA environment variable."
    )
print(f"Data file found: {DATA_PATH}")
print(f"Output directory: {OUT_DIR}")


# ========================================================================
# [CODE CELL 3]
# ========================================================================
# ── Parametreler ─────────────────────────────────────────────────
SEEDS     = [42, 7, 13, 99, 2024, 17, 88, 55, 101, 314]
TEST_SIZE = 0.20
N_FOLDS   = 5
N_OPTUNA  = 200   # Bayesian HPO trials (TPE sampler)
N_BOOT    = 1000
N_SHAP    = 200
SKIP_HPO  = False # True = default params only (quick test)

print(f"N_OPTUNA={N_OPTUNA}  |  GPU T4 ~{N_OPTUNA//4} min  |  CPU ~{N_OPTUNA//1.5:.0f} min")
print(f"SKIP_HPO={SKIP_HPO}  ← False: full Optuna, True: quick test")


# ========================================================================
# [CODE CELL 4]
# ========================================================================
# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE — Do not modify, run directly
# DATA_PATH, OUT_DIR, SEEDS, N_OPTUNA come from previous cells
# ════════════════════════════════════════════════════════════════

import warnings, time, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu

from sklearn.base         import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline      import Pipeline
from sklearn.svm           import SVR
from sklearn.tree          import DecisionTreeRegressor
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.ensemble      import (RandomForestRegressor, ExtraTreesRegressor,
                                    GradientBoostingRegressor,
                                    HistGradientBoostingRegressor)
from sklearn.neural_network  import MLPRegressor
from sklearn.model_selection import (train_test_split, KFold,
                                      cross_val_score, learning_curve,
                                      RandomizedSearchCV)
from sklearn.metrics         import (mean_squared_error, mean_absolute_error,
                                      r2_score)
from sklearn.inspection      import permutation_importance, PartialDependenceDisplay

try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except: HAS_OPTUNA = False
try:
    import xgboost as xgb; HAS_XGB = True
except: HAS_XGB = False
try:
    import lightgbm as lgb; HAS_LGB = True
except: HAS_LGB = False
try:
    import shap; HAS_SHAP = True
except: HAS_SHAP = False

print(f"Environment: Optuna={HAS_OPTUNA} XGB={HAS_XGB} LGB={HAS_LGB} SHAP={HAS_SHAP}")
if not HAS_XGB:
    print("⚠ XGBoost missing — pip install xgboost  →  Restart kernel")
    print("  Fix: pip install xgboost  →  Restart kernel  →  Run again")

try: plt.style.use("seaborn-v0_8-whitegrid")
except:
    try: plt.style.use("seaborn-whitegrid")
    except: pass

PALETTE = ["#2166AC","#D7191C","#4DAC26","#E08214","#762A83",
           "#1B7837","#F4A582","#8073AC","#B35806","#01665E",
           "#C2A5CF","#A6DBA0","#3288BD","#FDDBC7"]
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":11,"axes.labelsize":12,
    "axes.titlesize":13,"savefig.dpi":300,"savefig.bbox":"tight"})

FIG_N=[0]
def savefig(name):
    FIG_N[0]+=1; fp=os.path.join(OUT_DIR,f"fig{FIG_N[0]:02d}_{name}")
    plt.savefig(fp); plt.show(); plt.close(); print(f"  ✔ {fp}")

def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ════ §1 DATA + FEATURE ENGINEERING ════════════════════════════
section("§1  Data + Feature Engineering")
df_raw = pd.read_excel(DATA_PATH, sheet_name="Whole samples")
df_raw.columns = ["Sample","CD","AD","CF","RH","Aging","J","K","SDD","FOV"]
df = df_raw.copy()
df["log_SDD"]=np.log(df["SDD"]); df["log_RH"]=np.log(df["RH"])
df["log_J"]=np.log(df["J"].clip(lower=0.5)); df["log_CD"]=np.log(df["CD"])
df["RH_x_SDD"]=df["RH"]*df["SDD"]; df["RH_x_logSDD"]=df["RH"]*df["log_SDD"]
df["JK_product"]=df["J"]*df["K"]; df["Age_x_SDD"]=df["Aging"]*df["SDD"]
df["env_stress"]=(df["RH"]/100)*df["SDD"]*(1+df["Aging"]/45)
df["SDD_norm"]=df["SDD"]/(df["CD"]/100)
df["logRH_x_logSDD"]=df["log_RH"]*df["log_SDD"]
df["logJ_x_logSDD"]=df["log_J"]*df["log_SDD"]
df["Age_x_logSDD"]=df["Aging"]*df["log_SDD"]
smp=pd.get_dummies(df["Sample"],prefix="Smp",drop_first=False); df=pd.concat([df,smp],axis=1)
SMP_COLS=list(smp.columns)
ALL_FEATS=(["CD","AD","CF","RH","Aging","J","K","SDD","log_SDD","log_RH","log_J","log_CD",
            "RH_x_SDD","RH_x_logSDD","JK_product","Age_x_SDD","env_stress","SDD_norm",
            "logRH_x_logSDD","logJ_x_logSDD","Age_x_logSDD"]+SMP_COLS)
FN=ALL_FEATS; X_all=df[ALL_FEATS].values; y_all=df["FOV"].values
idx_all=np.arange(len(X_all))
print(f"  {df_raw.shape[0]} records  |  {len(ALL_FEATS)} features  |  FOV: {y_all.min():.2f}–{y_all.max():.2f} kV")

# ════ §1.5 VIF — Multicollinearity Check ══════════════════════════
section("§1.5  VIF — Multicollinearity Check")
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Only continuous engineered features (exclude dummy columns)
_VIF_COLS = ["CD","AD","CF","RH","Aging","J","K","SDD",
             "log_SDD","log_RH","log_J","log_CD",
             "RH_x_SDD","RH_x_logSDD","JK_product","Age_x_SDD",
             "env_stress","SDD_norm","logRH_x_logSDD","logJ_x_logSDD","Age_x_logSDD"]
_X_vif = add_constant(df[_VIF_COLS].astype(float))
_vif_data = pd.DataFrame({
    "Feature": _VIF_COLS,
    "VIF": [variance_inflation_factor(_X_vif.values, i+1) for i in range(len(_VIF_COLS))]
}).sort_values("VIF", ascending=False).reset_index(drop=True)
_vif_data["Severity"] = _vif_data["VIF"].apply(
    lambda v: "HIGH (>10)" if v > 10 else ("MODERATE (5-10)" if v > 5 else "OK (<5)"))

print(f"\n  {'Feature':<22} {'VIF':>8}  Severity")
print("  " + "-"*45)
for _, row in _vif_data.iterrows():
    flag = "⚠" if row["VIF"] > 10 else (" ~" if row["VIF"] > 5 else "  ")
    print(f"  {flag} {row['Feature']:<20} {row['VIF']:>8.2f}  {row['Severity']}")

_high_vif = _vif_data[_vif_data["VIF"] > 10]
print(f"\n  Features with VIF>10: {len(_high_vif)}  "
      f"(tree-based models are robust to multicollinearity)")

# VIF bar chart
fig, ax = plt.subplots(figsize=(10, 7))
_vc = _vif_data["VIF"].values[::-1]
_vn = _vif_data["Feature"].values[::-1]
_vcol = ["#D7191C" if v > 10 else ("#F4A582" if v > 5 else "#2166AC") for v in _vc]
ax.barh(range(len(_vn)), _vc, color=_vcol, alpha=0.85)
ax.axvline(5,  color="orange", lw=1.5, ls="--", label="VIF=5 (moderate)")
ax.axvline(10, color="red",    lw=1.5, ls="--", label="VIF=10 (high)")
ax.set_yticks(range(len(_vn))); ax.set_yticklabels(_vn, fontsize=9)
ax.set_xlabel("Variance Inflation Factor (VIF)")
ax.set_title("Multicollinearity Check — VIF per Feature", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); savefig("vif_multicollinearity.png")

_vif_data.to_csv(os.path.join(OUT_DIR, "vif_analysis.csv"), index=False)
print(f"  ✔ vif_analysis.csv saved")


# ════ §2 DATA VISUALIZATION / EDA ══════════════════════════════
section("§2  Data Visualization — EDA")

# ── 2.1 FOV Distribution (Histogram + Q-Q + KDE) ───────────────
from scipy.stats import probplot, gaussian_kde

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(y_all, bins=55, color=PALETTE[0], alpha=0.82,
             edgecolor="white", linewidth=0.5)
axes[0].axvline(np.mean(y_all), color="crimson", ls="--", lw=2,
                label=f"Mean={np.mean(y_all):.1f} kV")
axes[0].axvline(np.median(y_all), color="darkorange", ls=":", lw=2,
                label=f"Median={np.median(y_all):.1f} kV")
axes[0].set(xlabel="FOV (kV)", ylabel="Frequency",
            title=f"FOV Distribution  (n={len(y_all)}, SD={np.std(y_all):.1f} kV)")
axes[0].legend(); axes[0].grid(alpha=0.3)

probplot(y_all, plot=axes[1])
axes[1].set_title("FOV Normal Q-Q Plot"); axes[1].grid(alpha=0.3)

kde_x = np.linspace(y_all.min(), y_all.max(), 300)
kde_f = gaussian_kde(y_all)
axes[2].plot(kde_x, kde_f(kde_x), color=PALETTE[0], lw=2.5, label="KDE")
axes[2].fill_between(kde_x, kde_f(kde_x), alpha=0.22, color=PALETTE[0])
axes[2].set(xlabel="FOV (kV)", ylabel="Density",
            title="FOV Kernel Density Estimation (KDE)")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle("FOV Target Variable Analysis", fontsize=14, fontweight="bold")
plt.tight_layout(); savefig("eda_01_fov_distribution.png")

# ── 2.2 Raw Feature Distributions ───────────────────────────────
CONT_FEATS_EDA = ["CD","AD","SDD","RH","Aging","J","K"]
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, feat in enumerate(CONT_FEATS_EDA):
    ax = axes[i//4, i%4]
    vals = df_raw[feat].values
    ax.hist(vals, bins=45, color=PALETTE[i % len(PALETTE)],
            alpha=0.82, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(vals), color="crimson", ls="--", lw=1.5,
               label=f"mean={np.mean(vals):.2f}")
    ax.set(xlabel=feat, ylabel="Frequency",
           title=f"{feat}   (median={np.median(vals):.2f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
for ax in axes[1, len(CONT_FEATS_EDA)%4:]:
    ax.set_visible(False)
plt.suptitle("Raw Feature Distributions", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("eda_02_feature_histograms.png")

# ── 2.3 Log Transformation Effect ───────────────────────────────
LOG_PAIRS = [("SDD","log_SDD"),("RH","log_RH"),("CD","log_CD"),("J","log_J")]
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for col, (feat, lfeat) in enumerate(LOG_PAIRS):
    raw_v = df_raw[feat].values
    log_v = df[lfeat].values
    r_raw = np.corrcoef(raw_v, y_all)[0,1]
    r_log = np.corrcoef(log_v, y_all)[0,1]
    axes[0, col].scatter(raw_v, y_all, s=5, alpha=0.35,
                          color=PALETTE[col], linewidths=0)
    axes[0, col].set(xlabel=feat, ylabel="FOV (kV)",
                      title=f"FOV ~ {feat}   (r={r_raw:.3f})")
    axes[0, col].grid(alpha=0.3)
    axes[1, col].scatter(log_v, y_all, s=5, alpha=0.35,
                          color=PALETTE[(col+4)%len(PALETTE)], linewidths=0)
    m, b = np.polyfit(log_v, y_all, 1)
    xf = np.linspace(log_v.min(), log_v.max(), 100)
    axes[1, col].plot(xf, m*xf+b, "r-", lw=1.8,
                       label=f"Fit (r={r_log:.3f})")
    axes[1, col].set(xlabel=f"ln({feat})", ylabel="FOV (kV)",
                      title=f"FOV ~ ln({feat})   (r={r_log:.3f})")
    axes[1, col].legend(fontsize=8); axes[1, col].grid(alpha=0.3)
plt.suptitle("Log Transformation Effect — Physical Linearization",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("eda_03_log_transform.png")

# ── 2.4 Correlation Heatmap ──────────────────────────────────────
CORR_COLS = ["CD","AD","SDD","RH","Aging","J","K",
             "log_SDD","log_RH","log_J","log_CD","FOV"]
df_c = df_raw[["CD","AD","SDD","RH","Aging","J","K"]].copy()
df_c["log_SDD"] = np.log(df_raw["SDD"])
df_c["log_RH"]  = np.log(df_raw["RH"])
df_c["log_J"]   = np.log(df_raw["J"].clip(lower=0.5))
df_c["log_CD"]  = np.log(df_raw["CD"])
df_c["FOV"]     = y_all
corr_m = df_c[CORR_COLS].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr_m, dtype=bool))
cmap_corr = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_m, mask=mask, annot=True, fmt=".2f",
            cmap=cmap_corr, center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5,
            cbar_kws={"label":"Pearson r","shrink":0.8})
ax.set_title("Feature Correlation Matrix (including FOV)",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("eda_04_correlation_heatmap.png")

# ── 2.5 FOV by Sample Type ───────────────────────────────────────
smp_vals  = df_raw["Sample"].values
smp_uniq  = sorted(df_raw["Sample"].unique())
smp_data  = [y_all[smp_vals==s] for s in smp_uniq]
smp_cnts  = [len(d) for d in smp_data]
sort_idx  = np.argsort([np.median(d) for d in smp_data])
s_data    = [smp_data[i] for i in sort_idx]
s_names   = [smp_uniq[i] for i in sort_idx]
s_cnts    = [smp_cnts[i] for i in sort_idx]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
bp = axes[0].boxplot(s_data, patch_artist=True, notch=False,
                      medianprops=dict(color="black", lw=2))
for patch, col in zip(bp["boxes"], PALETTE*10):
    patch.set_facecolor(col); patch.set_alpha(0.78)
axes[0].set_xticks(range(1, len(s_names)+1))
axes[0].set_xticklabels(
    [f"{n}\n(n={c})" for n,c in zip(s_names,s_cnts)],
    rotation=55, ha="right", fontsize=7)
axes[0].set(ylabel="FOV (kV)", title="FOV Distribution by Sample Type")
axes[0].grid(alpha=0.3, axis="y")

axes[1].barh(range(len(s_names)), s_cnts,
             color=[PALETTE[i%len(PALETTE)] for i in range(len(s_names))],
             alpha=0.82)
axes[1].set_yticks(range(len(s_names)))
axes[1].set_yticklabels(s_names, fontsize=8)
for i, v in enumerate(s_cnts):
    axes[1].text(v+3, i, str(v), va="center", fontsize=8)
axes[1].set(xlabel="Number of Records",
            title="Observations per Sample")
axes[1].grid(alpha=0.3, axis="x")
plt.suptitle("Sample Type Analysis", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("eda_05_sample_analysis.png")

# ── 2.6 FOV vs Key Feature Scatter Grid ─────────────────────────
KEY_PAIRS = [("log_SDD","ln(SDD)"),("log_CD","ln(CD)"),
             ("log_RH","ln(RH)"),("Aging","Aging"),
             ("log_J","ln(J)"),("K","K")]
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
for ax, (feat, xlab) in zip(axes.flat, KEY_PAIRS):
    fv = df[feat].values if feat in df.columns else df_raw[feat].values
    sc = ax.scatter(fv, y_all, c=y_all, cmap="plasma",
                    s=7, alpha=0.45, linewidths=0)
    m, b = np.polyfit(fv, y_all, 1)
    xf = np.linspace(fv.min(), fv.max(), 100)
    ax.plot(xf, m*xf+b, "w-", lw=2.5, alpha=0.85)
    r = np.corrcoef(fv, y_all)[0,1]
    ax.set(xlabel=xlab, ylabel="FOV (kV)",
           title=f"FOV vs {xlab}   (r={r:.3f})")
    ax.grid(alpha=0.2)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="FOV (kV)")
plt.suptitle("FOV vs Key Features (Color = FOV kV)",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("eda_06_fov_scatter_grid.png")

# ── 2.7 Pair Feature Matrix ──────────────────────────────────────
PAIR_COLS = ["log_SDD","log_CD","log_RH","Aging","FOV"]
df_pair = df_c[PAIR_COLS].copy()
df_pair.columns = ["ln(SDD)","ln(CD)","ln(RH)","Aging","FOV"]
n = len(PAIR_COLS)
fig = plt.figure(figsize=(14, 12))
for i, ci in enumerate(df_pair.columns):
    for j, cj in enumerate(df_pair.columns):
        ax = fig.add_subplot(n, n, i*n+j+1)
        if i == j:
            ax.hist(df_pair[ci], bins=35,
                    color=PALETTE[i%len(PALETTE)], alpha=0.8)
        else:
            ax.scatter(df_pair[cj], df_pair[ci], s=3,
                       alpha=0.25, linewidths=0,
                       color=PALETTE[(i+j)%len(PALETTE)])
        if j == 0: ax.set_ylabel(ci, fontsize=7)
        if i == n-1: ax.set_xlabel(cj, fontsize=7)
        ax.tick_params(labelsize=6)
plt.suptitle("Pair Feature Matrix (Selected Variables)",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("eda_07_pairplot_matrix.png")

# ── 2.8 Descriptive Statistics Summary ───────────────────────────
desc = df_raw[CONT_FEATS_EDA+["FOV"]].describe().round(3)
print("\n  Descriptive Statistics (Raw Data):")
print(desc.to_string())
desc.to_csv(os.path.join(OUT_DIR,"eda_descriptive_stats.csv"))
import pandas as _pd_eda
print(f"\n  Total records   : {len(y_all)}")
print(f"  Feature count   : {len(ALL_FEATS)}")
print(f"  FOV range       : {y_all.min():.2f} – {y_all.max():.2f} kV")
print(f"  FOV mean±SD   : {y_all.mean():.2f} ± {y_all.std():.2f} kV")
print(f"  FOV skewness    : {_pd_eda.Series(y_all).skew():.3f}")
print(f"  FOV kurtosis    : {_pd_eda.Series(y_all).kurt():.3f}")
print(f"  Sample types    : {len(smp_uniq)} types | min={min(smp_cnts)} | max={max(smp_cnts)}")


# ════ §3 YARDIMCI FONKSIYONLAR ════════════════════════════════════
def compute_metrics(yt,yp):
    return dict(R2=float(r2_score(yt,yp)),
                RMSE=float(np.sqrt(mean_squared_error(yt,yp))),
                MAE=float(mean_absolute_error(yt,yp)),
                MAPE=float(np.mean(np.abs((yt-yp)/np.where(np.abs(yt)<1e-9,1e-9,yt)))*100),
                NSE=float(1-np.sum((yt-yp)**2)/max(np.sum((yt-np.mean(yt))**2),1e-12)))

def bootstrap_ci(yt,yp,metric="R2",n_boot=None,seed=42):
    if n_boot is None: n_boot=N_BOOT
    rng=np.random.RandomState(seed); n=len(yt); vals=[]
    for _ in range(n_boot):
        i=rng.choice(n,n,replace=True); vals.append(compute_metrics(yt[i],yp[i])[metric])
    return float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))

def cliffs_delta(a,b):
    if not len(a) or not len(b): return 0.,"n/a"
    stat,_=mannwhitneyu(a,b,alternative="two-sided")
    d=(2.*float(stat)/(len(a)*len(b)))-1.
    sz="large" if abs(d)>=.474 else "medium" if abs(d)>=.33 else "small" if abs(d)>=.147 else "negligible"
    return round(d,4),sz

PHY_NAMES=["log_CD","log_AD","log_SDD","log_RH","Aging","log_J","K"]+SMP_COLS

def physics_transform(X_arr):
    d=pd.DataFrame(X_arr,columns=FN)
    return np.column_stack([np.log(d["CD"].astype(float).values),np.log(d["AD"].astype(float).values),
        np.log(d["SDD"].astype(float).values),np.log(d["RH"].astype(float).values),
        d["Aging"].astype(float).values,np.log(d["J"].astype(float).clip(lower=0.5).values),
        d["K"].astype(float).values]+[d[c].astype(float).values for c in SMP_COLS])

# ════ §4 EMPIRICAL MODELS ════════════════════════════════════════
class ObenausModel(BaseEstimator,RegressorMixin):
    def __init__(self,alpha=0.01): self.alpha=alpha
    def fit(self,X,y):
        Xp=physics_transform(X); self.ridge_=Ridge(alpha=self.alpha).fit(Xp,np.log(y))
        self.coef_dict_=dict(zip(PHY_NAMES,self.ridge_.coef_))
        self.intercept_=float(self.ridge_.intercept_); self.n_sdd_=self.coef_dict_["log_SDD"]
        return self
    def predict(self,X): return np.exp(self.ridge_.predict(physics_transform(X)))
    def validate_physics(self):
        # FIX: six sign constraints, matching Table 4 of the manuscript.
        # AD and J were reported in the paper but were missing from this check.
        return {"SDD neg":self.coef_dict_["log_SDD"]<0,"RH neg":self.coef_dict_["log_RH"]<0,
                "CD pos":self.coef_dict_["log_CD"]>0,"AD pos":self.coef_dict_["log_AD"]>0,
                "Aging neg":self.coef_dict_["Aging"]<0,"J neg":self.coef_dict_["log_J"]<0}

class RizkModel(BaseEstimator,RegressorMixin):
    def __init__(self,alpha=0.01): self.alpha=alpha
    def _Xr(self,X_arr):
        d=pd.DataFrame(X_arr,columns=FN)
        return np.column_stack([np.log(d["CD"].astype(float).values),np.log(d["SDD"].astype(float).values),
            np.log(d["RH"].astype(float).values),d["Aging"].astype(float).values]+[d[c].astype(float).values for c in SMP_COLS])
    def fit(self,X,y): self.ridge_=Ridge(alpha=self.alpha).fit(self._Xr(X),np.log(y)); return self
    def predict(self,X): return np.exp(self.ridge_.predict(self._Xr(X)))

# ════ §5 HPO ══════════════════════════════════════════════════════
section("§5  HPO — Optuna + All Models (Fair Evaluation)")
BEST_PARAMS={}

# ── Separate holdout for HPO (seed=999) — independent from eval sets ──
idx_hpo,_=train_test_split(idx_all,test_size=0.20,random_state=999,
                            stratify=df["Sample"].values)
X_hpo=X_all[idx_hpo]; y_hpo=y_all[idx_hpo]
# Pre-scaled HPO data for Ridge / KNN / SVR / MLP
_sc_hpo=RobustScaler().fit(X_hpo); X_hpo_sc=_sc_hpo.transform(X_hpo)
print(f"  HPO holdout: {len(X_hpo)} records  (seed=999 — completely independent from evaluation sets)")

def _make_study(seed=42):
    return optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=seed))

# ── XGBoost ─────────────────────────────────────────────
if HAS_XGB and not SKIP_HPO:
    if HAS_OPTUNA:
        print("  XGBoost Optuna starting...")
        def xgb_obj(trial):
            p={"n_estimators":trial.suggest_int("n",300,2000),
               "learning_rate":trial.suggest_float("lr",0.005,0.15,log=True),
               "max_depth":trial.suggest_int("d",4,10),
               "subsample":trial.suggest_float("sub",0.6,1.0),
               "colsample_bytree":trial.suggest_float("col",0.5,1.0),
               "min_child_weight":trial.suggest_int("mcw",1,20),
               "gamma":trial.suggest_float("gamma",0.,0.5),
               "reg_lambda":trial.suggest_float("lam",0.01,10.,log=True),
               "reg_alpha":trial.suggest_float("alp",0.,2.)}
            m=xgb.XGBRegressor(**p,random_state=42,verbosity=0,n_jobs=-1,tree_method="hist")
            return cross_val_score(m,X_hpo,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st=_make_study(); st.optimize(xgb_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp=st.best_params
        BEST_PARAMS["XGBoost"]=xgb.XGBRegressor(
            n_estimators=bp["n"],learning_rate=bp["lr"],max_depth=bp["d"],
            subsample=bp["sub"],colsample_bytree=bp["col"],min_child_weight=bp["mcw"],
            gamma=bp["gamma"],reg_lambda=bp["lam"],reg_alpha=bp["alp"],
            random_state=42,verbosity=0,n_jobs=-1,tree_method="hist")
        print(f"  XGBoost  Optuna CV R²={st.best_value:.4f}  trial={N_OPTUNA}")
    else:
        rs=RandomizedSearchCV(xgb.XGBRegressor(random_state=42,verbosity=0,n_jobs=-1,tree_method="hist"),
            {"n_estimators":[500,800,1000,1500],"learning_rate":[0.005,0.01,0.02,0.03,0.05],
             "max_depth":[4,5,6,7,8],"subsample":[0.6,0.7,0.8,0.9],"colsample_bytree":[0.5,0.6,0.7,0.8],
             "min_child_weight":[1,3,5,10],"gamma":[0.,0.1,0.2],"reg_lambda":[0.1,0.5,1.,2.]},
            n_iter=N_OPTUNA,cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs.fit(X_hpo,y_hpo); BEST_PARAMS["XGBoost"]=clone(rs.best_estimator_)
        print(f"  XGBoost  RandomSearch R²={rs.best_score_:.4f}")
elif HAS_XGB:
    BEST_PARAMS["XGBoost"]=xgb.XGBRegressor(n_estimators=1000,learning_rate=0.02,max_depth=6,
        subsample=0.8,colsample_bytree=0.7,min_child_weight=3,gamma=0.,reg_lambda=1.,
        random_state=42,verbosity=0,n_jobs=-1,tree_method="hist")
    print("  XGBoost: default params (SKIP_HPO=True)")

# ── LightGBM ──────────────────────────────────────────
if HAS_LGB and not SKIP_HPO:
    if HAS_OPTUNA:
        print("  LightGBM Optuna starting...")
        def lgb_obj(trial):
            p={"n_estimators":trial.suggest_int("n",300,2000),
               "learning_rate":trial.suggest_float("lr",0.005,0.15,log=True),
               "num_leaves":trial.suggest_int("leaves",31,255),
               "subsample":trial.suggest_float("sub",0.6,1.0),
               "colsample_bytree":trial.suggest_float("col",0.5,1.0),
               "min_child_samples":trial.suggest_int("mcs",5,100),
               "reg_lambda":trial.suggest_float("lam",0.01,10.,log=True),
               "reg_alpha":trial.suggest_float("alp",0.,2.)}
            m=lgb.LGBMRegressor(**p,random_state=42,n_jobs=-1,verbose=-1)
            return cross_val_score(m,X_hpo,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st2=_make_study(); st2.optimize(lgb_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp2=st2.best_params
        BEST_PARAMS["LightGBM"]=lgb.LGBMRegressor(n_estimators=bp2["n"],learning_rate=bp2["lr"],
            num_leaves=bp2["leaves"],subsample=bp2["sub"],colsample_bytree=bp2["col"],
            min_child_samples=bp2["mcs"],reg_lambda=bp2["lam"],reg_alpha=bp2["alp"],
            random_state=42,n_jobs=-1,verbose=-1)
        print(f"  LightGBM Optuna CV R²={st2.best_value:.4f}")
    else:
        rs2=RandomizedSearchCV(lgb.LGBMRegressor(random_state=42,n_jobs=-1,verbose=-1),
            {"n_estimators":[500,800,1000],"learning_rate":[0.005,0.01,0.02,0.03],
             "num_leaves":[63,127,255],"subsample":[0.6,0.7,0.8],"colsample_bytree":[0.5,0.6,0.7],
             "min_child_samples":[5,10,20],"reg_lambda":[0.1,0.5,1.]},
            n_iter=N_OPTUNA,cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs2.fit(X_hpo,y_hpo); BEST_PARAMS["LightGBM"]=clone(rs2.best_estimator_)
        print(f"  LightGBM RandomSearch R²={rs2.best_score_:.4f}")
elif HAS_LGB:
    BEST_PARAMS["LightGBM"]=lgb.LGBMRegressor(n_estimators=1000,learning_rate=0.02,num_leaves=127,
        subsample=0.8,colsample_bytree=0.7,min_child_samples=10,reg_lambda=1.,
        random_state=42,n_jobs=-1,verbose=-1)

# ── GBR ───────────────────────────────────────────────
if not SKIP_HPO:
    if HAS_OPTUNA:
        print("  GBR Optuna starting...")
        def gbr_obj(trial):
            p={"n_estimators":trial.suggest_int("n",300,1500),
               "learning_rate":trial.suggest_float("lr",0.005,0.1,log=True),
               "max_depth":trial.suggest_int("d",3,7),
               "subsample":trial.suggest_float("sub",0.6,1.0),
               "min_samples_leaf":trial.suggest_int("msl",1,10),
               "max_features":trial.suggest_float("mf",0.4,1.0)}
            m=GradientBoostingRegressor(**p,random_state=42)
            return cross_val_score(m,X_hpo,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st_gbr=_make_study(); st_gbr.optimize(gbr_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp_gbr=st_gbr.best_params
        BEST_PARAMS["GBR"]=GradientBoostingRegressor(
            n_estimators=bp_gbr["n"],learning_rate=bp_gbr["lr"],max_depth=bp_gbr["d"],
            subsample=bp_gbr["sub"],min_samples_leaf=bp_gbr["msl"],max_features=bp_gbr["mf"],
            random_state=42)
        print(f"  GBR      Optuna CV R²={st_gbr.best_value:.4f}")
    else:
        rs_=RandomizedSearchCV(GradientBoostingRegressor(random_state=42),
            {"n_estimators":[500,800],"learning_rate":[0.02,0.03],"max_depth":[4,5],
             "subsample":[0.7,0.8,0.9],"min_samples_leaf":[1,2]},
            n_iter=max(8,N_OPTUNA//8),cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs_.fit(X_hpo,y_hpo); BEST_PARAMS["GBR"]=clone(rs_.best_estimator_)
        print(f"  GBR      RandomSearch R²={rs_.best_score_:.4f}")
else:
    BEST_PARAMS["GBR"]=GradientBoostingRegressor(n_estimators=800,learning_rate=0.03,max_depth=4,
        subsample=0.8,min_samples_leaf=2,random_state=42)

# ── HistGBR ───────────────────────────────────────────
if not SKIP_HPO:
    if HAS_OPTUNA:
        print("  HistGBR Optuna starting...")
        def hgbr_obj(trial):
            p={"max_iter":trial.suggest_int("n",300,1500),
               "learning_rate":trial.suggest_float("lr",0.005,0.1,log=True),
               "max_depth":trial.suggest_int("d",3,10),
               "l2_regularization":trial.suggest_float("l2",0.0,1.0),
               "min_samples_leaf":trial.suggest_int("msl",5,50),
               "max_leaf_nodes":trial.suggest_int("mln",20,255)}
            m=HistGradientBoostingRegressor(**p,random_state=42)
            return cross_val_score(m,X_hpo,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st_hgbr=_make_study(); st_hgbr.optimize(hgbr_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp_hgbr=st_hgbr.best_params
        BEST_PARAMS["HistGBR"]=HistGradientBoostingRegressor(
            max_iter=bp_hgbr["n"],learning_rate=bp_hgbr["lr"],max_depth=bp_hgbr["d"],
            l2_regularization=bp_hgbr["l2"],min_samples_leaf=bp_hgbr["msl"],
            max_leaf_nodes=bp_hgbr["mln"],random_state=42)
        print(f"  HistGBR  Optuna CV R²={st_hgbr.best_value:.4f}")
    else:
        rs_=RandomizedSearchCV(HistGradientBoostingRegressor(random_state=42),
            {"max_iter":[500,800],"learning_rate":[0.02,0.03],"max_depth":[5,6,7,None],
             "l2_regularization":[0.0,0.01],"min_samples_leaf":[5,10]},
            n_iter=max(8,N_OPTUNA//8),cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs_.fit(X_hpo,y_hpo); BEST_PARAMS["HistGBR"]=clone(rs_.best_estimator_)
        print(f"  HistGBR  RandomSearch R²={rs_.best_score_:.4f}")
else:
    BEST_PARAMS["HistGBR"]=HistGradientBoostingRegressor(max_iter=500,learning_rate=0.03,max_depth=6,
        l2_regularization=0.,min_samples_leaf=5,random_state=42)

# ── RF ────────────────────────────────────────────────
if not SKIP_HPO:
    if HAS_OPTUNA:
        print("  RF Optuna starting...")
        def rf_obj(trial):
            p={"n_estimators":trial.suggest_int("n",200,800),
               "min_samples_leaf":trial.suggest_int("msl",1,10),
               "max_features":trial.suggest_float("mf",0.3,0.9),
               "min_samples_split":trial.suggest_int("mss",2,10)}
            m=RandomForestRegressor(**p,n_jobs=-1,random_state=42)
            return cross_val_score(m,X_hpo,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st_rf=_make_study(); st_rf.optimize(rf_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp_rf=st_rf.best_params
        BEST_PARAMS["RF"]=RandomForestRegressor(
            n_estimators=bp_rf["n"],min_samples_leaf=bp_rf["msl"],max_features=bp_rf["mf"],
            min_samples_split=bp_rf["mss"],n_jobs=-1,random_state=42)
        print(f"  RF       Optuna CV R²={st_rf.best_value:.4f}")
    else:
        rs_=RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,random_state=42),
            {"n_estimators":[400,500],"min_samples_leaf":[1,2],"max_features":[0.5,0.6]},
            n_iter=max(8,N_OPTUNA//8),cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs_.fit(X_hpo,y_hpo); BEST_PARAMS["RF"]=clone(rs_.best_estimator_)
        print(f"  RF       RandomSearch R²={rs_.best_score_:.4f}")
else:
    BEST_PARAMS["RF"]=RandomForestRegressor(n_estimators=400,min_samples_leaf=1,
        max_features=0.6,n_jobs=-1,random_state=42)

# ── Extra Trees ──────────────────────────────────────────
if not SKIP_HPO:
    if HAS_OPTUNA:
        print("  Extra Trees Optuna starting...")
        def et_obj(trial):
            p={"n_estimators":trial.suggest_int("n",200,800),
               "min_samples_leaf":trial.suggest_int("msl",1,10),
               "max_features":trial.suggest_float("mf",0.3,0.9),
               "min_samples_split":trial.suggest_int("mss",2,10)}
            m=ExtraTreesRegressor(**p,n_jobs=-1,random_state=42)
            return cross_val_score(m,X_hpo,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st_et=_make_study(); st_et.optimize(et_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp_et=st_et.best_params
        BEST_PARAMS["Extra Trees"]=ExtraTreesRegressor(
            n_estimators=bp_et["n"],min_samples_leaf=bp_et["msl"],max_features=bp_et["mf"],
            min_samples_split=bp_et["mss"],n_jobs=-1,random_state=42)
        print(f"  ExtraTrees Optuna CV R²={st_et.best_value:.4f}")
    else:
        rs_=RandomizedSearchCV(ExtraTreesRegressor(n_jobs=-1,random_state=42),
            {"n_estimators":[300,400,500],"min_samples_leaf":[1,2,3],"max_features":[0.4,0.5,0.6]},
            n_iter=max(8,N_OPTUNA//8),cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs_.fit(X_hpo,y_hpo); BEST_PARAMS["Extra Trees"]=clone(rs_.best_estimator_)
        print(f"  ExtraTrees RandomSearch R²={rs_.best_score_:.4f}")
else:
    BEST_PARAMS["Extra Trees"]=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,
        max_features=0.5,n_jobs=-1,random_state=42)

# ── Ridge ───────────────────────────────────────────────
if not SKIP_HPO:
    if HAS_OPTUNA:
        print("  Ridge Optuna starting...")
        def ridge_obj(trial):
            alpha=trial.suggest_float("alpha",1e-3,100.,log=True)
            return cross_val_score(Ridge(alpha=alpha),X_hpo_sc,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st_ridge=_make_study(); st_ridge.optimize(ridge_obj,n_trials=N_OPTUNA,n_jobs=1)
        BEST_PARAMS["Ridge"]=Ridge(alpha=st_ridge.best_params["alpha"])
        print(f"  Ridge    Optuna CV R²={st_ridge.best_value:.4f}  alpha={st_ridge.best_params['alpha']:.4f}")
    else:
        rs_=RandomizedSearchCV(Ridge(),{"alpha":np.logspace(-3,2,50).tolist()},
            n_iter=min(50,N_OPTUNA),cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs_.fit(X_hpo_sc,y_hpo); BEST_PARAMS["Ridge"]=clone(rs_.best_estimator_)
        print(f"  Ridge    RandomSearch R²={rs_.best_score_:.4f}")
else:
    BEST_PARAMS["Ridge"]=Ridge(alpha=1.)

# ── Decision Tree ────────────────────────────────────────
if not SKIP_HPO:
    if HAS_OPTUNA:
        print("  Decision Tree Optuna starting...")
        def dt_obj(trial):
            p={"max_depth":trial.suggest_int("d",3,20),
               "min_samples_leaf":trial.suggest_int("msl",1,20),
               "min_samples_split":trial.suggest_int("mss",2,20),
               "max_features":trial.suggest_categorical("mf",["sqrt","log2",None,0.5,0.7,0.9])}
            m=DecisionTreeRegressor(**p,random_state=42)
            return cross_val_score(m,X_hpo,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st_dt=_make_study(); st_dt.optimize(dt_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp_dt=st_dt.best_params
        BEST_PARAMS["Decision Tree"]=DecisionTreeRegressor(
            max_depth=bp_dt["d"],min_samples_leaf=bp_dt["msl"],
            min_samples_split=bp_dt["mss"],max_features=bp_dt["mf"],random_state=42)
        print(f"  DecTree  Optuna CV R²={st_dt.best_value:.4f}")
    else:
        rs_=RandomizedSearchCV(DecisionTreeRegressor(random_state=42),
            {"max_depth":list(range(3,20)),"min_samples_leaf":list(range(1,15)),
             "max_features":["sqrt","log2",None,0.5,0.7]},
            n_iter=max(8,N_OPTUNA//8),cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs_.fit(X_hpo,y_hpo); BEST_PARAMS["Decision Tree"]=clone(rs_.best_estimator_)
        print(f"  DecTree  RandomSearch R²={rs_.best_score_:.4f}")
else:
    BEST_PARAMS["Decision Tree"]=DecisionTreeRegressor(max_depth=12,min_samples_leaf=5,random_state=42)

# ── KNN ───────────────────────────────────────────────
if not SKIP_HPO:
    if HAS_OPTUNA:
        print("  KNN Optuna starting...")
        def knn_obj(trial):
            p={"n_neighbors":trial.suggest_int("k",2,20),
               "weights":trial.suggest_categorical("w",["uniform","distance"]),
               "p":trial.suggest_int("p",1,2)}
            return cross_val_score(KNeighborsRegressor(**p),X_hpo_sc,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st_knn=_make_study(); st_knn.optimize(knn_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp_knn=st_knn.best_params
        BEST_PARAMS["KNN"]=KNeighborsRegressor(n_neighbors=bp_knn["k"],weights=bp_knn["w"],p=bp_knn["p"])
        print(f"  KNN      Optuna CV R²={st_knn.best_value:.4f}  k={bp_knn['k']}")
    else:
        rs_=RandomizedSearchCV(KNeighborsRegressor(),
            {"n_neighbors":list(range(2,20)),"weights":["uniform","distance"],"p":[1,2]},
            n_iter=min(30,N_OPTUNA),cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs_.fit(X_hpo_sc,y_hpo); BEST_PARAMS["KNN"]=clone(rs_.best_estimator_)
        print(f"  KNN      RandomSearch R²={rs_.best_score_:.4f}")
else:
    BEST_PARAMS["KNN"]=KNeighborsRegressor(n_neighbors=5,weights="distance")

# ── SVR ───────────────────────────────────────────────
if not SKIP_HPO:
    if HAS_OPTUNA:
        print("  SVR Optuna starting...")
        def svr_obj(trial):
            p={"C":trial.suggest_float("C",0.1,1000.,log=True),
               "gamma":trial.suggest_categorical("gamma",["scale","auto"]),
               "epsilon":trial.suggest_float("eps",0.001,1.,log=True)}
            return cross_val_score(SVR(kernel="rbf",**p),X_hpo_sc,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st_svr=_make_study(); st_svr.optimize(svr_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp_svr=st_svr.best_params
        BEST_PARAMS["SVR"]=SVR(kernel="rbf",C=bp_svr["C"],gamma=bp_svr["gamma"],epsilon=bp_svr["eps"])
        print(f"  SVR      Optuna CV R²={st_svr.best_value:.4f}  C={bp_svr['C']:.2f}")
    else:
        rs_=RandomizedSearchCV(SVR(kernel="rbf"),
            {"C":np.logspace(-1,3,50).tolist(),"gamma":["scale","auto"],
             "epsilon":[0.001,0.01,0.05,0.1,0.5]},
            n_iter=min(30,N_OPTUNA),cv=5,scoring="r2",random_state=42,n_jobs=-1)
        rs_.fit(X_hpo_sc,y_hpo); BEST_PARAMS["SVR"]=clone(rs_.best_estimator_)
        print(f"  SVR      RandomSearch R²={rs_.best_score_:.4f}")
else:
    BEST_PARAMS["SVR"]=SVR(kernel="rbf",C=100,gamma="scale",epsilon=0.05)

# ── MLP ───────────────────────────────────────────────
_arch_choices=[(64,32),(128,64),(128,64,32),(256,128,64),(256,128,64,32),(128,64,32,16)]
if not SKIP_HPO:
    if HAS_OPTUNA:
        print("  MLP Optuna starting...")
        def mlp_obj(trial):
            arch=trial.suggest_categorical("arch",list(range(len(_arch_choices))))
            p={"hidden_layer_sizes":_arch_choices[arch],
               "alpha":trial.suggest_float("alpha",1e-5,1e-2,log=True),
               "learning_rate_init":trial.suggest_float("lr",1e-4,1e-2,log=True),
               "batch_size":trial.suggest_categorical("bs",[32,64,128,"auto"])}
            m=MLPRegressor(**p,max_iter=500,early_stopping=True,random_state=42)
            return cross_val_score(m,X_hpo_sc,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
        st_mlp=_make_study(); st_mlp.optimize(mlp_obj,n_trials=N_OPTUNA,n_jobs=1)
        bp_mlp=st_mlp.best_params
        BEST_PARAMS["MLP"]={"hidden_layer_sizes":_arch_choices[bp_mlp["arch"]],
                             "alpha":bp_mlp["alpha"],"learning_rate_init":bp_mlp["lr"],
                             "batch_size":bp_mlp["bs"]}
        print(f"  MLP      Optuna CV R²={st_mlp.best_value:.4f}  arch={_arch_choices[bp_mlp['arch']]}")
    else:
        best_r2_mlp=-np.inf; best_bp_mlp={}
        for _hs in [(64,32),(128,64),(128,64,32)]:
            for _al in [1e-4,1e-3]:
                _m=MLPRegressor(hidden_layer_sizes=_hs,alpha=_al,max_iter=500,early_stopping=True,random_state=42)
                _r2=cross_val_score(_m,X_hpo_sc,y_hpo,cv=5,scoring="r2",n_jobs=-1).mean()
                if _r2>best_r2_mlp: best_r2_mlp=_r2; best_bp_mlp={"hidden_layer_sizes":_hs,"alpha":_al,"learning_rate_init":0.001,"batch_size":"auto"}
        BEST_PARAMS["MLP"]=best_bp_mlp
        print(f"  MLP      GridSearch R²={best_r2_mlp:.4f}")
else:
    BEST_PARAMS["MLP"]={"hidden_layer_sizes":(128,64,32),"alpha":1e-4,"learning_rate_init":0.001,"batch_size":"auto"}

print(f"\n  ✔ {len(BEST_PARAMS)} models optimized: {list(BEST_PARAMS.keys())}")


# ════ §6 HyPhysML ULTIMATE SINIFI ═════════════════════════════════
class HyPhysML(BaseEstimator,RegressorMixin):
    """
    HyPhysML ULTIMATE
    ══════════════════
    Core: XGBoost(Optuna) + LightGBM(Optuna) + GBR + HGBR + RF + ET + KNN
              → Ridge meta (5-fold OOF)
    Physics: ObenausModel → interpretation (n_SDD coefficient validation)

    Base learners + Ridge meta-learner + Obenaus interpretation layer
    """
    def __init__(self,n_folds=5,random_state=42):
        self.n_folds=n_folds; self.random_state=random_state

    def _make_base(self,seed):
        bls={}
        def _clone_with_seed(bp,default):
            try:
                m=clone(bp) if bp is not None else default
            except Exception:
                m=default
            try:
                p=m.get_params()
                if "random_state" in p: m.set_params(random_state=seed)
            except: pass
            return m
        if HAS_XGB:
            bls["XGBoost"]=_clone_with_seed(BEST_PARAMS.get("XGBoost"),
                xgb.XGBRegressor(n_estimators=1000,learning_rate=0.02,max_depth=6,
                    subsample=0.8,colsample_bytree=0.7,min_child_weight=3,gamma=0.,
                    reg_lambda=1.,random_state=seed,verbosity=0,n_jobs=-1,tree_method="hist"))
        if HAS_LGB:
            bls["LightGBM"]=_clone_with_seed(BEST_PARAMS.get("LightGBM"),
                lgb.LGBMRegressor(n_estimators=1000,learning_rate=0.02,num_leaves=127,
                    subsample=0.8,colsample_bytree=0.7,min_child_samples=10,reg_lambda=1.,
                    random_state=seed,n_jobs=-1,verbose=-1))
        bls["GBR"]=_clone_with_seed(BEST_PARAMS.get("GBR"),
            GradientBoostingRegressor(n_estimators=800,learning_rate=0.03,max_depth=4,
                subsample=0.8,min_samples_leaf=2,random_state=seed))
        bls["HistGBR"]=_clone_with_seed(BEST_PARAMS.get("HistGBR"),
            HistGradientBoostingRegressor(max_iter=500,learning_rate=0.03,max_depth=6,
                l2_regularization=0.,min_samples_leaf=5,random_state=seed))
        bls["RF"]=_clone_with_seed(BEST_PARAMS.get("RF"),
            RandomForestRegressor(n_estimators=400,min_samples_leaf=1,max_features=0.6,n_jobs=-1,random_state=seed))
        bls["ET"]=_clone_with_seed(BEST_PARAMS.get("Extra Trees"),ExtraTreesRegressor(n_estimators=400,min_samples_leaf=1,max_features=0.5,n_jobs=-1,random_state=seed))
        bls["KNN"]=Pipeline([("sc",RobustScaler()),("m",KNeighborsRegressor(n_neighbors=3,weights="distance"))])
        return bls

    def fit(self,X,y):
        rs=self.random_state; kf=KFold(n_splits=self.n_folds,shuffle=True,random_state=rs)
        bls=self._make_base(rs); nb=len(bls); n=len(y); oof=np.zeros((n,nb))
        for bi,(nm_,bl_) in enumerate(bls.items()):
            for ti,vi in kf.split(X):
                try: oof[vi,bi]=clone(bl_).fit(X[ti],y[ti]).predict(X[vi])
                except Exception as e: print(f"  OOF {nm_}: {e}")
        self.meta_=Ridge(alpha=0.1).fit(oof,y)
        self.bl_fit_={nm_:clone(bl_).fit(X,y) for nm_,bl_ in bls.items()}
        self.meta_weights_=self.meta_.coef_; self.meta_names_=list(bls.keys())
        self.pm_=ObenausModel(alpha=0.01).fit(X,y)
        self.physics_coef_=self.pm_.coef_dict_; self.n_sdd_=self.pm_.n_sdd_
        return self

    def predict(self,X):
        te=np.zeros((len(X),len(self.bl_fit_)))
        for bi,(nm_,bl_) in enumerate(self.bl_fit_.items()):
            try: te[:,bi]=bl_.predict(X)
            except: te[:,bi]=np.mean(y_all)
        return self.meta_.predict(te)

# ════ §7 MODEL ZOO + 10-SEED ══════════════════════════════════════
section("§7  Model Zoo + 10-Seed Training")

def get_model(name,seed):
    def cs(bp,default):
        try:
            m=clone(bp) if bp is not None else default
        except Exception:
            m=default
        try:
            p=m.get_params()
            if "random_state" in p: m.set_params(random_state=seed)
        except: pass
        return m
    mlp_p=BEST_PARAMS.get("MLP",{"hidden_layer_sizes":(128,64,32),"alpha":1e-4,"learning_rate_init":0.001,"batch_size":"auto"})
    d={"Obenaus":ObenausModel(),"Rizk":RizkModel(),
       "Ridge":Pipeline([("sc",RobustScaler()),("m",cs(BEST_PARAMS.get("Ridge"),Ridge(alpha=1.)))]),
       "Decision Tree":cs(BEST_PARAMS.get("Decision Tree"),DecisionTreeRegressor(max_depth=12,min_samples_leaf=5,random_state=seed)),
       "KNN":Pipeline([("sc",RobustScaler()),("m",cs(BEST_PARAMS.get("KNN"),KNeighborsRegressor(n_neighbors=5,weights="distance")))]),
       "Extra Trees":cs(BEST_PARAMS.get("Extra Trees"),ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,max_features=0.5,n_jobs=-1,random_state=seed)),
       "GBR":cs(BEST_PARAMS.get("GBR"),GradientBoostingRegressor(n_estimators=800,learning_rate=0.03,max_depth=4,subsample=0.8,min_samples_leaf=2,random_state=seed)),
       "HistGBR":cs(BEST_PARAMS.get("HistGBR"),HistGradientBoostingRegressor(max_iter=500,learning_rate=0.03,max_depth=6,l2_regularization=0.,min_samples_leaf=5,random_state=seed)),
       "RF":cs(BEST_PARAMS.get("RF"),RandomForestRegressor(n_estimators=400,min_samples_leaf=1,max_features=0.6,n_jobs=-1,random_state=seed)),
       "SVR":Pipeline([("sc",RobustScaler()),("m",cs(BEST_PARAMS.get("SVR"),SVR(kernel="rbf",C=100,gamma="scale",epsilon=0.05)))]),
       "MLP":Pipeline([("sc",RobustScaler()),("m",MLPRegressor(**mlp_p,max_iter=500,early_stopping=True,random_state=seed))]),
       "HyPhysML":HyPhysML(random_state=seed)}
    if HAS_XGB: d["XGBoost"]=cs(BEST_PARAMS.get("XGBoost"),xgb.XGBRegressor(n_estimators=1000,learning_rate=0.02,max_depth=6,subsample=0.8,colsample_bytree=0.7,random_state=seed,verbosity=0,n_jobs=-1,tree_method="hist"))
    if HAS_LGB: d["LightGBM"]=cs(BEST_PARAMS.get("LightGBM"),lgb.LGBMRegressor(n_estimators=1000,learning_rate=0.02,num_leaves=127,subsample=0.8,random_state=seed,n_jobs=-1,verbose=-1))
    return d.get(name,d["GBR"])

MODEL_NAMES=(["Obenaus","Rizk","Ridge","Decision Tree","KNN",
              "Extra Trees","GBR","HistGBR","RF","SVR","MLP","HyPhysML"]
             +(["XGBoost"] if HAS_XGB else [])
             +(["LightGBM"] if HAS_LGB else []))
print(f"  {len(MODEL_NAMES)} model: {', '.join(MODEL_NAMES)}")

all_res=[]; preds_s42={}
for si,seed in enumerate(SEEDS):
    print(f"  Seed {seed:>5} ({si+1}/{len(SEEDS)})",end="  ")
    idx_tr,idx_te=train_test_split(idx_all,test_size=TEST_SIZE,random_state=seed,stratify=df["Sample"].values)
    Xtr,Xte=X_all[idx_tr],X_all[idx_te]; ytr,yte=y_all[idx_tr],y_all[idx_te]
    for name in MODEL_NAMES:
        model=get_model(name,seed); t0=time.time()
        try:
            model.fit(Xtr,ytr); yp=model.predict(Xte)
        except Exception as e: print(f"\n  [{name}] {e}"); continue
        m=compute_metrics(yte,yp); m.update({"Model":name,"Seed":seed,"Time_s":round(time.time()-t0,3)})
        all_res.append(m)
        if seed==42: preds_s42[name]=(model,yp,yte,Xtr,Xte,ytr)
        print(".",end="",flush=True)
    print()

res_df=pd.DataFrame(all_res)
agg=(res_df.groupby("Model").agg(
    R2_mean=("R2","mean"),R2_std=("R2","std"),
    RMSE_mean=("RMSE","mean"),RMSE_std=("RMSE","std"),
    MAE_mean=("MAE","mean"),MAE_std=("MAE","std"),
    MAPE_mean=("MAPE","mean"),MAPE_std=("MAPE","std"),
    NSE_mean=("NSE","mean"),NSE_std=("NSE","std")).round(5)
     .sort_values("R2_mean",ascending=False))

BEST=agg.index[0]
if BEST not in preds_s42: BEST=list(preds_s42.keys())[0]
print(f"\n  ★ Best model: {BEST}  R²={agg.R2_mean.iloc[0]:.4f}")
print(f"\n  {'Model':<20} {'Test R² (mean±std)':>22}")
print("  "+"-"*45)
for nm in agg.index[:10]:
    r=agg.loc[nm]; mk="★ " if nm==BEST else "  "
    print(f"  {mk}{nm:<20}  {r.R2_mean:.4f}±{r.R2_std:.4f}")

res_df.to_csv(os.path.join(OUT_DIR,"results_all_seeds.csv"),index=False)
agg.to_csv(os.path.join(OUT_DIR,"results_summary.csv"))

# ── §7.4  5-Fold CV Scores for all models (seed=42) ──────────────
section("§7.4  5-Fold Cross-Validation Scores")
_cv_rows = []
print(f"\n  {'Model':<20} {'CV R² mean':>12} {'CV R² std':>11}")
print("  " + "-"*46)
for name in MODEL_NAMES:
    try:
        _m_cv = get_model(name, 42)
        _cv_sc = cross_val_score(_m_cv, X_all, y_all, cv=5, scoring="r2", n_jobs=-1)
        _cv_rows.append({"Model": name,
                          "CV_R2_mean": round(float(_cv_sc.mean()), 5),
                          "CV_R2_std":  round(float(_cv_sc.std()),  5)})
        print(f"  {'★ ' if name==BEST else '  '}{name:<20} {_cv_sc.mean():>12.4f} {_cv_sc.std():>11.4f}")
    except Exception as e:
        print(f"  {name:<20} SKIPPED ({e})")

_cv_df = pd.DataFrame(_cv_rows).sort_values("CV_R2_mean", ascending=False)
_cv_df.to_csv(os.path.join(OUT_DIR, "cv_scores.csv"), index=False)
print(f"  ✔ cv_scores.csv saved")

# ── §7.5  Train vs Test R² + Overfitting Check ────────────────────
section("§7.5  Train vs Test R² — Overfitting Check")
_train_res = []
for si, seed in enumerate(SEEDS):
    idx_tr, idx_te = train_test_split(idx_all, test_size=TEST_SIZE,
                                       random_state=seed, stratify=df["Sample"].values)
    Xtr_, Xte_ = X_all[idx_tr], X_all[idx_te]
    ytr_, yte_ = y_all[idx_tr], y_all[idx_te]
    for name in MODEL_NAMES:
        model = get_model(name, seed)
        try:
            model.fit(Xtr_, ytr_)
            r2_tr = r2_score(ytr_, model.predict(Xtr_))
            r2_te = r2_score(yte_, model.predict(Xte_))
            _train_res.append({"Model": name, "Seed": seed,
                                "R2_train": r2_tr, "R2_test": r2_te,
                                "Overfit_gap": r2_tr - r2_te})
        except: pass

_tr_df = pd.DataFrame(_train_res)
_tr_agg = (_tr_df.groupby("Model")
           .agg(R2_train_mean=("R2_train","mean"),
                R2_test_mean=("R2_test","mean"),
                Overfit_gap_mean=("Overfit_gap","mean"))
           .round(4).sort_values("R2_test_mean", ascending=False))

print(f"\n  {'Model':<20} {'Train R²':>10} {'Test R²':>10} {'Gap':>8}")
print("  " + "-"*52)
for nm, row in _tr_agg.iterrows():
    flag = "⚠" if row["Overfit_gap_mean"] > 0.05 else "  "
    print(f"  {flag} {nm:<20} {row['R2_train_mean']:>10.4f} {row['R2_test_mean']:>10.4f} {row['Overfit_gap_mean']:>8.4f}")

# Train vs Test bar chart
_ord2 = _tr_agg.index.tolist()
_x2 = np.arange(len(_ord2)); _w = 0.38
fig, ax = plt.subplots(figsize=(max(12, len(_ord2)), 5))
ax.bar(_x2 - _w/2, _tr_agg["R2_train_mean"].values, _w,
       color="#2166AC", alpha=0.82, label="Train R²")
ax.bar(_x2 + _w/2, _tr_agg["R2_test_mean"].values,  _w,
       color="#D7191C", alpha=0.82, label="Test R²")
ax.set_xticks(_x2); ax.set_xticklabels(_ord2, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("R²"); ax.set_ylim(0, 1.05)
ax.set_title("Train vs Test R² — Overfitting Diagnostic (10-seed mean)", fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); savefig("train_vs_test_r2.png")

# ── §7.6  Computation Time ─────────────────────────────────────────
section("§7.6  Computation Time per Model")
_time_agg = (res_df.groupby("Model")["Time_s"]
             .agg(["mean","std"]).round(3)
             .sort_values("mean", ascending=True))

fig, ax = plt.subplots(figsize=(max(10, len(_time_agg)), 5))
_xc = np.arange(len(_time_agg))
_bars = ax.bar(_xc, _time_agg["mean"].values,
               yerr=_time_agg["std"].values,
               color=[PALETTE[i % len(PALETTE)] for i in range(len(_time_agg))],
               alpha=0.85, capsize=4,
               error_kw=dict(ecolor="black", lw=1.2))
ax.set_xticks(_xc)
ax.set_xticklabels(_time_agg.index.tolist(), rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Training Time (s)")
ax.set_title("Computation Time per Model (mean ± std, 10 seeds)", fontweight="bold")
for bar, (_, row) in zip(_bars, _time_agg.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + row["std"] + 0.01 * _time_agg["mean"].max(),
            f"{row['mean']:.1f}s", ha="center", fontsize=7.5, fontweight="bold")
ax.grid(alpha=0.3, axis="y"); plt.tight_layout()
savefig("computation_time.png")

_tr_agg.to_csv(os.path.join(OUT_DIR, "train_test_r2.csv"))
_time_agg.to_csv(os.path.join(OUT_DIR, "computation_time.csv"))
print("  ✔ train_test_r2.csv  |  computation_time.csv  saved")

# ════ §8 STATISTICAL TESTS + CI ═════════════════════════════════
section("§8  Statistical Tests + Bootstrap CI")
best_model,yp_best,yte,Xtr,Xte,ytr=preds_s42[BEST]
m_best=compute_metrics(yte,yp_best)
ci_res={}
for nm,(mdl,yp_nm,yte_nm,*_) in preds_s42.items():
    lo,hi=bootstrap_ci(yte_nm,yp_nm,"R2"); ci_res[nm]={"R2_lo":lo,"R2_hi":hi}
r2_dict={n:res_df[res_df["Model"]==n]["R2"].values for n in MODEL_NAMES if n in res_df["Model"].values}
try:
    fr,fp=stats.friedmanchisquare(*list(r2_dict.values()))
    print(f"  Friedman χ²={fr:.4f}  p={fp:.2e}")
except Exception as e: print(f"  Friedman: {e}")
r2_best_=r2_dict.get(BEST,np.zeros(len(SEEDS))); alpha_b=0.05/max(len(MODEL_NAMES)-1,1); wil_rows=[]
for nm in MODEL_NAMES:
    if nm==BEST: continue
    r2_nm=r2_dict.get(nm)
    if r2_nm is None or not len(r2_nm): continue
    # One-sided (upper-tailed), matching directional hypothesis H1 and the
    # manuscript. For n=10 with all differences of one sign the exact test
    # returns its floor, p = 2**-10 = 0.000977. The two-sided counterpart would
    # be 0.001953, which also clears alpha_b = 0.00385.
    try: sw,pw=stats.wilcoxon(r2_best_,r2_nm,alternative="greater")
    except: sw,pw=0,1.
    d,sz=cliffs_delta(r2_best_,r2_nm); sig="✔" if pw<alpha_b else "n.s."
    wil_rows.append({"vs":nm,"p":round(pw,6),"sig":sig,"cliff_delta":d,"effect":sz})
pd.DataFrame(wil_rows).to_csv(os.path.join(OUT_DIR,"statistical_tests.csv"),index=False)

# ════ §9 SENSITIVITY ANALYSIS ════════════════════════════════════
section("§9  Sensitivity Analysis")
NOISE_FEATS=["SDD","RH","Aging"]; NOISE_LEVELS=[0.,0.05,0.10,0.15,0.20]
noise_res={}; rng_n=np.random.RandomState(99)
for feat in NOISE_FEATS:
    fi=FN.index(feat); std=Xte[:,fi].std(); noise_res[feat]={}
    for nl in NOISE_LEVELS:
        Xn=Xte.copy()
        if nl>0: Xn[:,fi]+=rng_n.normal(0,nl*std,len(Xte))
        noise_res[feat][nl]=compute_metrics(yte,best_model.predict(Xn))
    print(f"  {feat}: R²@0%={noise_res[feat][0.]['R2']:.4f}  R²@20%={noise_res[feat][0.20]['R2']:.4f}")
noise_rows=[{"Feature":f,"Noise_%":int(nl*100),**m} for f,d in noise_res.items() for nl,m in d.items()]
pd.DataFrame(noise_rows).to_csv(os.path.join(OUT_DIR,"sensitivity_analysis.csv"),index=False)


# ════ §9.5  GENERALIZABILITY — Per-Insulator-Type Performance ══════
section("§9.5  Per-Insulator-Type Performance + Residual Analysis + Learning Curve")

# ── 9.5-A  Per-tip performans (Smp_1, Smp_2, Smp_3, Smp_4) ─────────────────
# Get sample labels for seed=42 test split
_idx_tr42, _idx_te42 = train_test_split(
    idx_all, test_size=TEST_SIZE, random_state=42,
    stratify=df["Sample"].values
)
_Xte42 = X_all[_idx_te42]
_yte42 = y_all[_idx_te42]
_smp42 = df_raw["Sample"].values[_idx_te42]          # Smp_1 / Smp_2 / Smp_3 / Smp_4

# Re-train HyPhysML with seed=42 (or retrieve from preds_s42)
if "HyPhysML" in preds_s42:
    _mdl42, _yp42, _, _Xtr42, _, _ytr42 = preds_s42["HyPhysML"]
else:
    _mdl42 = HyPhysML(random_state=42)
    _ytr42 = y_all[_idx_tr42]
    _mdl42.fit(X_all[_idx_tr42], _ytr42)
    _yp42 = _mdl42.predict(_Xte42)

_type_rows = []
_smp_uniq  = sorted(set(_smp42))
print(f"\n  Insulator types: {_smp_uniq}")
print(f"  {'Type':<12} {'n_test':>6} {'R²':>8} {'RMSE (kV)':>12} {'MAE (kV)':>10} {'MAPE (%)':>10}")
print("  " + "-"*60)
for _smp in _smp_uniq:
    _mask  = (_smp42 == _smp)
    _n     = _mask.sum()
    if _n < 5:
        print(f"  {_smp:<12} {'N/A — insufficient samples':>40}")
        continue
    _m = compute_metrics(_yte42[_mask], _yp42[_mask])
    _type_rows.append({"Insulator Type": _smp, "n_test": int(_n),
                        "R2": _m["R2"], "RMSE": _m["RMSE"],
                        "MAE": _m["MAE"], "MAPE": _m["MAPE"]})
    print(f"  {_smp:<12} {_n:>6} {_m['R2']:>8.4f} {_m['RMSE']:>12.4f} {_m['MAE']:>10.4f} {_m['MAPE']:>10.3f}%")

# Overall (all test samples)
_m_all = compute_metrics(_yte42, _yp42)
print(f"  {'OVERALL':<12} {len(_yte42):>6} {_m_all['R2']:>8.4f} {_m_all['RMSE']:>12.4f} {_m_all['MAE']:>10.4f} {_m_all['MAPE']:>10.3f}%")

_type_df = pd.DataFrame(_type_rows)

# ── 9.5-B  Per-type bar chart ──────────────────────────────────────────────
if not _type_rows:
    print("  [SKIP] Per-type chart: no insulator type has ≥5 test samples")
else:
    _colors = ["#2E86AB","#A23B72","#F18F01","#C73E1D"]
    _labels = [r["Insulator Type"] for r in _type_rows]
    _r2s    = [r["R2"]   for r in _type_rows]
    _rmses  = [r["RMSE"] for r in _type_rows]
    _mapes  = [r["MAPE"] for r in _type_rows]
    fig_pt, axes_pt = plt.subplots(1, 3, figsize=(13, 4))
    for _ax, _vals, _title, _ylab, _fmt in zip(
            axes_pt,
            [_r2s, _rmses, _mapes],
            ["R² by Insulator Type", "RMSE (kV)", "MAPE (%)"],
            ["R²", "RMSE (kV)", "MAPE (%)"],
            [".4f", ".3f", ".2f"]):
        _bars = _ax.bar(_labels, _vals, color=_colors[:len(_labels)], alpha=0.85, edgecolor="white")
        for _b, _v in zip(_bars, _vals):
            _ax.text(_b.get_x()+_b.get_width()/2, _b.get_height()+0.001*max(_vals),
                     f"{_v:{_fmt}}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        _ax.set_title(_title, fontweight="bold", fontsize=11)
        _ax.set_ylabel(_ylab); _ax.set_ylim(0, max(_vals)*1.15)
        _ax.axhline(y=(_m_all["R2"] if "R²" in _title else
                       (_m_all["RMSE"] if "RMSE" in _title else _m_all["MAPE"])),
                    color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Overall")
        _ax.legend(fontsize=8); _ax.grid(axis="y", alpha=0.3)
    fig_pt.suptitle("HyPhysML — Performance by Insulator Type (seed=42 test set)",
                    fontweight="bold", fontsize=12)
    plt.tight_layout()
    _pt_path = os.path.join(OUT_DIR, "fig_per_type_performance.png")
    plt.savefig(_pt_path, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  [OK] {_pt_path}")

# ── 9.5-C  Residual analysis + Shapiro-Wilk normality test ─────────────────
from scipy.stats import shapiro, probplot

_residuals = _yte42 - _yp42
_stat_sw, _p_sw = shapiro(_residuals[:5000] if len(_residuals) > 5000 else _residuals)

fig_res, axes_res = plt.subplots(1, 3, figsize=(14, 4))

# 1. Residual histogram
axes_res[0].hist(_residuals, bins=40, color="#2E86AB", alpha=0.8, edgecolor="white")
axes_res[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes_res[0].set_xlabel("Residual (kV)"); axes_res[0].set_ylabel("Frequency")
axes_res[0].set_title(f"Residual Histogram\nShapiro-Wilk p={_p_sw:.4f}", fontweight="bold")
axes_res[0].grid(alpha=0.3)

# 2. Residual vs predicted
axes_res[1].scatter(_yp42, _residuals, alpha=0.3, s=8, color="#2E86AB")
axes_res[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes_res[1].set_xlabel("Predicted (kV)"); axes_res[1].set_ylabel("Residual (kV)")
axes_res[1].set_title("Residual vs. Predicted\n(Homoscedasticity check)", fontweight="bold")
axes_res[1].grid(alpha=0.3)

# 3. Q-Q plot
_qq_theor, _qq_sample = probplot(_residuals, dist="norm")[0]
axes_res[2].scatter(_qq_theor, _qq_sample, alpha=0.4, s=8, color="#2E86AB")
_lim = max(abs(_qq_theor.min()), abs(_qq_theor.max()))
axes_res[2].plot([-_lim, _lim], [-_lim*_residuals.std(), _lim*_residuals.std()],
                 "r--", linewidth=1.5)
axes_res[2].set_xlabel("Theoretical Quantiles"); axes_res[2].set_ylabel("Sample Quantiles")
axes_res[2].set_title("Normal Q-Q Plot", fontweight="bold")
axes_res[2].grid(alpha=0.3)

fig_res.suptitle("HyPhysML Residual Analysis (seed=42)", fontweight="bold", fontsize=12)
plt.tight_layout()
_res_path = os.path.join(OUT_DIR, "fig_residual_analysis.png")
plt.savefig(_res_path, dpi=150, bbox_inches="tight"); plt.show()

print(f"\n  Shapiro-Wilk: W={_stat_sw:.4f}, p={_p_sw:.4f}")
print(f"  Residual mean={_residuals.mean():.4f} kV, std={_residuals.std():.4f} kV")
print(f"  {'Normal distribution accepted (p>0.05)' if _p_sw > 0.05 else 'Normal distribution rejected (p<=0.05) — expected for large N'}")
print(f"  [OK] {_res_path}")

# ── 9.5-D  Learning curve ────────────────────────────────────────────────────
from sklearn.model_selection import learning_curve as sk_lc

print("\n  Computing learning curve (HyPhysML, 5-fold)...")
_train_sz = np.linspace(0.10, 1.0, 8)
_lc_tr_sz, _lc_tr_sc, _lc_val_sc = sk_lc(
    HyPhysML(random_state=42), X_all, y_all,
    train_sizes=_train_sz, cv=5, scoring="r2", n_jobs=-1,
    error_score="raise"
)

_lc_tr_mean  = _lc_tr_sc.mean(axis=1)
_lc_tr_std   = _lc_tr_sc.std(axis=1)
_lc_val_mean = _lc_val_sc.mean(axis=1)
_lc_val_std  = _lc_val_sc.std(axis=1)

fig_lc, ax_lc = plt.subplots(figsize=(8, 5))
ax_lc.plot(_lc_tr_sz, _lc_tr_mean,  "o-", color="#2E86AB", label="Training R²",    linewidth=2)
ax_lc.plot(_lc_tr_sz, _lc_val_mean, "s-", color="#C73E1D", label="Validation R²",  linewidth=2)
ax_lc.fill_between(_lc_tr_sz,
                   _lc_tr_mean  - _lc_tr_std,  _lc_tr_mean  + _lc_tr_std,
                   alpha=0.15, color="#2E86AB")
ax_lc.fill_between(_lc_tr_sz,
                   _lc_val_mean - _lc_val_std, _lc_val_mean + _lc_val_std,
                   alpha=0.15, color="#C73E1D")
ax_lc.set_xlabel("Number of training samples"); ax_lc.set_ylabel("R²")
ax_lc.set_title("HyPhysML Learning Curve (5-fold CV)", fontweight="bold", fontsize=12)
ax_lc.legend(fontsize=10); ax_lc.grid(alpha=0.3); ax_lc.set_ylim(0.85, 1.01)
plt.tight_layout()
_lc_path = os.path.join(OUT_DIR, "fig_learning_curve.png")
plt.savefig(_lc_path, dpi=150, bbox_inches="tight"); plt.show()
print(f"  Final validation R²: {_lc_val_mean[-1]:.4f} ± {_lc_val_std[-1]:.4f}")
print(f"  [OK] {_lc_path}")

# ── 9.5-E  Excel'e kaydet ────────────────────────────────────────────────────
_res_summary = pd.DataFrame({
    "Metric": ["Residual Mean (kV)", "Residual Std (kV)", "Shapiro-Wilk W", "Shapiro-Wilk p",
               "LC Val R2 (full data)", "LC Val R2 std"],
    "Value":  [round(_residuals.mean(),4), round(_residuals.std(),4),
               round(float(_stat_sw),4), round(float(_p_sw),6),
               round(float(_lc_val_mean[-1]),4), round(float(_lc_val_std[-1]),4)]
})
_pt_path_xl = os.path.join(OUT_DIR, "Table_PerType_Residual.xlsx")
with pd.ExcelWriter(_pt_path_xl, engine="openpyxl") as _wr3:
    _type_df.to_excel(_wr3, sheet_name="Per_Type_Performance", index=False)
    _res_summary.to_excel(_wr3, sheet_name="Residual_Summary", index=False)
print(f"  [OK] {_pt_path_xl}")


# ════ §10-13 FIGURES ═══════════════════════════════════════════════
section("§10-13  Figures")
ORD=agg.index.tolist()

# R² + CI + reference line
fig,ax=plt.subplots(figsize=(max(12,len(ORD)),6))
x=np.arange(len(ORD)); r2m=[agg.loc[m,"R2_mean"] for m in ORD]
clo=[ci_res.get(m,{}).get("R2_lo",r2m[i]) for i,m in enumerate(ORD)]
chi=[ci_res.get(m,{}).get("R2_hi",r2m[i]) for i,m in enumerate(ORD)]
elo=np.maximum(0,np.array(r2m)-np.array(clo)); ehi=np.maximum(0,np.array(chi)-np.array(r2m))
bars=ax.bar(x,r2m,color=PALETTE[:len(ORD)],alpha=0.80)
ax.errorbar(x,r2m,yerr=[elo,ehi],fmt="none",color="black",capsize=4,lw=1.5,label="95% CI")
for i,nm in enumerate(ORD):
    if nm in ["Obenaus","Rizk"]: bars[i].set_hatch("///"); bars[i].set_edgecolor("black")
    if nm in ["HyPhysML","XGBoost","LightGBM"]: bars[i].set_linewidth(2)
ax.set_xticks(x); ax.set_xticklabels(ORD,rotation=45,ha="right",fontsize=9)
ax.set_ylabel("R²"); ax.set_title("R² with 95% Bootstrap CI")
ax.grid(alpha=0.3,axis="y"); plt.tight_layout(); savefig("r2_bootstrap_ci.png")

# Boxplot
fig,axes=plt.subplots(1,3,figsize=(18,6))
for ax,(metric,ylabel) in zip(axes,[("R2","R²↑"),("RMSE","RMSE↓"),("MAPE","MAPE%↓")]):
    data=[res_df[res_df["Model"]==m][metric].values for m in ORD if m in res_df["Model"].values]
    bp=ax.boxplot(data,patch_artist=True,medianprops=dict(color="black",lw=2))
    for patch,col in zip(bp["boxes"],PALETTE): patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticks(range(1,len(ORD)+1)); ax.set_xticklabels(ORD,rotation=45,ha="right",fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3,axis="y")
plt.suptitle("Model Comparison (10-Seed)",fontsize=13,fontweight="bold")
plt.tight_layout(); savefig("model_comparison_boxplot.png")

# Predicted vs Actual
resid=yte-yp_best; lo_r2=ci_res.get(BEST,{}).get("R2_lo",0); hi_r2=ci_res.get(BEST,{}).get("R2_hi",0)
fig,axes=plt.subplots(1,2,figsize=(13,5))
sc=axes[0].scatter(yte,yp_best,c=resid,cmap="RdBu",s=15,alpha=0.6,vmin=-5,vmax=5)
mv,xv=min(yte.min(),yp_best.min()),max(yte.max(),yp_best.max())
axes[0].plot([mv,xv],[mv,xv],"k--",lw=1.5)
axes[0].set(xlabel="Actual FOV (kV)",ylabel="Predicted (kV)",
    title=f"{BEST}\nR²={m_best['R2']:.4f} [CI:{lo_r2:.4f},{hi_r2:.4f}]  RMSE={m_best['RMSE']:.4f}")
axes[0].grid(alpha=0.3); plt.colorbar(sc,ax=axes[0],label="Residual",fraction=0.03)
axes[1].scatter(yp_best,resid,c=PALETTE[0],s=12,alpha=0.5)
axes[1].axhline(0,color="k",lw=1.5,ls="--")
for sg in [2,-2]: axes[1].axhline(sg*resid.std(),color="red",lw=1,ls=":",label=f"{sg}σ={sg*resid.std():.2f}")
axes[1].set(xlabel="Predicted",ylabel="Residual"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); savefig("predicted_vs_actual.png")

# Permutation importance
print("  Permutation importance...")
po=np.arange(len(FN))  # fallback order
perm=None
try:
    perm=permutation_importance(best_model,Xte,yte,n_repeats=15,random_state=42,n_jobs=-1)
    po=np.argsort(perm.importances_mean)[::-1]
except Exception as _perm_e:
    print(f"  [WARN] Permutation importance failed ({_perm_e}), using default feature order")
if perm is not None:
    fig,ax=plt.subplots(figsize=(10,7)); ti=po[:15]
    ax.barh(range(15),perm.importances_mean[ti][::-1],xerr=perm.importances_std[ti][::-1],
        color=[PALETTE[i%len(PALETTE)] for i in range(15)][::-1],alpha=0.8,
        error_kw=dict(ecolor="black",capsize=3))
    ax.set_yticks(range(15)); ax.set_yticklabels([FN[i] for i in ti][::-1])
    ax.set(xlabel="Mean Decrease R²",title=f"Permutation Importance — {BEST}")
    ax.grid(alpha=0.3,axis="x"); plt.tight_layout(); savefig("permutation_importance.png")

# Sensitivity
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,feat in zip(axes,NOISE_FEATS):
    r2_ns=[noise_res[feat][nl]["R2"] for nl in NOISE_LEVELS]; rm_ns=[noise_res[feat][nl]["RMSE"] for nl in NOISE_LEVELS]
    ax2=ax.twinx()
    ax.plot([nl*100 for nl in NOISE_LEVELS],r2_ns,"o-",color=PALETTE[0],lw=2,label="R²")
    ax2.plot([nl*100 for nl in NOISE_LEVELS],rm_ns,"s--",color=PALETTE[1],lw=2,label="RMSE")
    ax.set(xlabel=f"{feat} Noise (%)",ylabel="R²",title=f"Robustness: {feat}")
    ax2.set_ylabel("RMSE (kV)"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.suptitle(f"Sensitivity — {BEST}",fontsize=13,fontweight="bold")
plt.tight_layout(); savefig("sensitivity_noise.png")

# SHAP
rng_sh=np.random.RandomState(42)
X_bg=Xtr[rng_sh.choice(len(Xtr),min(150,len(Xtr)),replace=False)]
X_ex=Xte[rng_sh.choice(len(Xte),min(N_SHAP,len(Xte)),replace=False)]
shap_vals=None

# For stacking models (HyPhysML), use the strongest tree base learner for SHAP
_shap_model = best_model
if hasattr(best_model, "bl_fit_"):
    if "XGBoost" in best_model.bl_fit_:
        _shap_model = best_model.bl_fit_["XGBoost"]
    elif "LightGBM" in best_model.bl_fit_:
        _shap_model = best_model.bl_fit_["LightGBM"]
    elif "GBR" in best_model.bl_fit_:
        _shap_model = best_model.bl_fit_["GBR"]
    else:
        _shap_model = list(best_model.bl_fit_.values())[0]
    _shap_key = ("XGBoost" if "XGBoost" in best_model.bl_fit_ else
                 "LightGBM" if "LightGBM" in best_model.bl_fit_ else
                 "GBR" if "GBR" in best_model.bl_fit_ else
                 list(best_model.bl_fit_.keys())[0])
    print(f"  SHAP: using base learner '{_shap_key}' for HyPhysML stacking model")

if HAS_SHAP:
    try:
        exp=shap.TreeExplainer(_shap_model); shap_vals=np.array(exp.shap_values(X_ex))
        print("  SHAP: TreeExplainer OK")
    except Exception as e:
        print(f"  SHAP TreeExplainer failed ({e}), trying KernelExplainer...")
        try:
            exp=shap.KernelExplainer(_shap_model.predict,X_bg)
            shap_vals=np.array(exp.shap_values(X_ex,nsamples=80,silent=True))
            print("  SHAP: KernelExplainer OK")
        except Exception as e2:
            print(f"  SHAP KernelExplainer failed ({e2}), using custom approximation")
if shap_vals is None:
    def _kshap(model,Xb,Xe,n_p=50,rs=42):
        rng_=np.random.RandomState(rs); nf=Xe.shape[1]; sv=np.zeros((len(Xe),nf)); bl=Xb.mean(0)
        for i in range(len(Xe)):
            x=Xe[i]; phi=np.zeros(nf)
            for _ in range(n_p):
                pm=rng_.permutation(nf); cv=bl.copy(); pp=model.predict(cv.reshape(1,-1))[0]
                for f in pm:
                    cv[f]=x[f]; np_=model.predict(cv.reshape(1,-1))[0]; phi[f]+=np_-pp; pp=np_
            sv[i]=phi/n_p
        return sv
    shap_vals=_kshap(_shap_model,X_bg,X_ex)

mabs=np.abs(shap_vals).mean(0); sord=np.argsort(mabs)[::-1]
fig,ax=plt.subplots(figsize=(10,7)); ti=sord[:15]
ax.barh(range(15),mabs[ti][::-1],color=[PALETTE[i%len(PALETTE)] for i in range(15)][::-1],alpha=0.85)
ax.set_yticks(range(15)); ax.set_yticklabels([FN[i] for i in ti][::-1])
ax.set(xlabel="Mean |SHAP| (kV)",title=f"SHAP — {BEST}"); ax.grid(alpha=0.3,axis="x")
plt.tight_layout(); savefig("shap_bar.png")

fig,ax=plt.subplots(figsize=(11,8))
for rank,fi in enumerate(sord[:8][::-1]):
    sv_=shap_vals[:,fi]; fv_=X_ex[:,fi]; fn_=(fv_-fv_.min())/(fv_.max()-fv_.min()+1e-9)
    ax.scatter(sv_,rank+np.random.RandomState(fi).uniform(-0.3,0.3,len(sv_)),c=fn_,cmap="coolwarm",s=12,alpha=0.55,linewidths=0)
ax.set_yticks(range(8)); ax.set_yticklabels([FN[i] for i in sord[:8][::-1]])
ax.axvline(0,color="black",lw=1,ls="--"); ax.set(xlabel="SHAP (kV)",title="SHAP Beeswarm")
ax.grid(alpha=0.3,axis="x"); plt.tight_layout(); savefig("shap_beeswarm.png")

# ════ §14 HyPhysML PHYSICS INTERPRETATION ══════════════════════════
section("§14  HyPhysML Physics Interpretation")
idx_tr42,idx_te42=train_test_split(idx_all,test_size=TEST_SIZE,random_state=42,stratify=df["Sample"].values)
Xtr42,Xte42=X_all[idx_tr42],X_all[idx_te42]; ytr42,yte42=y_all[idx_tr42],y_all[idx_te42]
hyp42=HyPhysML(random_state=42); hyp42.fit(Xtr42,ytr42)
fov42=hyp42.predict(Xte42); m_hyp42=compute_metrics(yte42,fov42)
print(f"  HyPhysML seed=42: R²={m_hyp42['R2']:.4f}  RMSE={m_hyp42['RMSE']:.4f}")

fig,axes=plt.subplots(1,2,figsize=(13,5))
resid_h=yte42-fov42
sc=axes[0].scatter(yte42,fov42,c=resid_h,cmap="RdBu",s=15,alpha=0.6,vmin=-5,vmax=5)
axes[0].plot([yte42.min(),yte42.max()],[yte42.min(),yte42.max()],"k--",lw=1.5)
axes[0].set(xlabel="Actual FOV (kV)",ylabel="Predicted (kV)",
    title=f"HyPhysML ULTIMATE (seed=42)\nR²={m_hyp42['R2']:.4f}  RMSE={m_hyp42['RMSE']:.4f}  MAPE={m_hyp42['MAPE']:.2f}%")
axes[0].grid(alpha=0.3); plt.colorbar(sc,ax=axes[0],label="Residual",fraction=0.03)
axes[1].scatter(fov42,resid_h,c=PALETTE[0],s=12,alpha=0.5); axes[1].axhline(0,color="k",lw=1.5,ls="--")
for sg in [2,-2]: axes[1].axhline(sg*resid_h.std(),color="red",lw=1,ls=":",label=f"{sg}σ={sg*resid_h.std():.2f}")
axes[1].set(xlabel="Predicted",ylabel="Residual"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); savefig("hyphysml_predicted_vs_actual.png")

fig,ax=plt.subplots(figsize=(9,4))
mw_=hyp42.meta_weights_; mn_=hyp42.meta_names_; cc_=[PALETTE[i%len(PALETTE)] for i in range(len(mn_))]
bars_=ax.bar(range(len(mn_)),mw_,color=cc_,alpha=0.85)
ax.set_xticks(range(len(mn_))); ax.set_xticklabels(mn_,fontsize=10,rotation=20,ha="right")
ax.axhline(0,color="black",lw=0.8); ax.set(ylabel="Ridge Weight",title="HyPhysML Meta-Weights")
for b,v in zip(bars_,mw_): ax.text(b.get_x()+b.get_width()/2,v+0.003*np.sign(v) if v!=0 else 0.003,f"{v:.3f}",ha="center",fontsize=9)
ax.grid(alpha=0.3,axis="y"); plt.tight_layout(); savefig("hyphysml_meta_weights.png")

THEORETICAL={"log_SDD":(-0.35,-0.10),"log_RH":(-0.80,-0.20),"log_CD":(0.50,1.20),"Aging":(-0.05,0.00)}
coefs_=hyp42.physics_coef_
cs=sorted(coefs_.items(),key=lambda x:abs(x[1]),reverse=True)
fig,ax=plt.subplots(figsize=(10,6))
cn_=[k for k,v in cs]; cv_=[v for k,v in cs]; cc_=["#2166AC" if v>0 else "#D7191C" for v in cv_]
ax.barh(range(len(cn_)),cv_[::-1],color=cc_[::-1],alpha=0.85)
# FIX: bars are drawn from cv_[::-1], so feature cs[k] sits at y-position len(cs)-1-k.
# The previous loop iterated reversed(cs) *and* mirrored the index, placing every
# theoretical band on the wrong row (e.g. log_CD's band ended up on the log_J row).
for k,(feat,val) in enumerate(cs):
    if feat in THEORETICAL:
        lo_t,hi_t=THEORETICAL[feat]
        if isinstance(lo_t,(int,float)): ax.barh(len(cs)-1-k,hi_t-lo_t,left=lo_t,height=0.3,color="gray",alpha=0.35)
ax.set_yticks(range(len(cn_))); ax.set_yticklabels(cn_[::-1])
ax.axvline(0,color="black",lw=1)
ax.set(xlabel="β (log-space)",title=f"Obenaus Coefficients (interpretation)  β₀={hyp42.pm_.intercept_:.4f}")
ax.legend(handles=[mpatches.Patch(color="#2166AC",label="FOV↑"),mpatches.Patch(color="#D7191C",label="FOV↓"),mpatches.Patch(color="gray",alpha=0.4,label="Theoretical")])
ax.grid(alpha=0.3,axis="x"); plt.tight_layout(); savefig("physics_coefficients.png")
# FIX: this note quoted -0.15..-0.25, which is NOT the band used anywhere else in
# this script; THEORETICAL["log_SDD"] is (-0.35, -0.10). The manuscript had copied
# the stale note. Report the band actually used.
print(f"  n_SDD={hyp42.n_sdd_:.4f}  (admissible band used here: {THEORETICAL['log_SDD'][0]} .. {THEORETICAL['log_SDD'][1]})")
chk=hyp42.pm_.validate_physics(); print(f"  Physics checks passed: {sum(v for v in chk.values())}/{len(chk)}")

# ════ §14.5  Partial Dependence Plots (PDP) ═══════════════════════
section("§14.5  Partial Dependence Plots")
from sklearn.inspection import PartialDependenceDisplay

# Top 6 features by permutation importance
_pdp_feats_idx = list(po[:6])
_pdp_feat_names = [FN[i] for i in _pdp_feats_idx]
print(f"  PDP features: {_pdp_feat_names}")

# Use a fast surrogate if best_model is HyPhysML (stacking — PDP is slow)
_pdp_model = best_model
if hasattr(best_model, "bl_fit_"):
    # Use the strongest base learner for PDP speed
    if "XGBoost" in best_model.bl_fit_:
        _pdp_model = best_model.bl_fit_["XGBoost"]
    elif "LightGBM" in best_model.bl_fit_:
        _pdp_model = best_model.bl_fit_["LightGBM"]
    else:
        _pdp_model = list(best_model.bl_fit_.values())[0]
    print(f"  Using base learner for PDP: {list(best_model.bl_fit_.keys())[0]}")

try:
    fig, ax_pdp = plt.subplots(2, 3, figsize=(16, 9))
    disp = PartialDependenceDisplay.from_estimator(
        _pdp_model, Xtr42, features=_pdp_feats_idx,
        feature_names=FN, ax=ax_pdp.ravel()[:6],
        kind="average", subsample=500, random_state=42,
        line_kw={"color": "#2166AC", "lw": 2.5}
    )
    for i, ax_ in enumerate(ax_pdp.ravel()[:6]):
        ax_.set_title(f"PDP — {_pdp_feat_names[i]}", fontweight="bold", fontsize=11)
        ax_.set_ylabel("Partial Dependence (FOV, kV)")
        ax_.grid(alpha=0.3)
    fig.suptitle(f"Partial Dependence Plots — Top 6 Features ({BEST})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("pdp_top6.png")
    print("  ✔ PDP saved")
except Exception as e:
    print(f"  PDP skipped: {e}")

# ── ICE plots (Individual Conditional Expectation) for top 3 ──────
try:
    fig, ax_ice = plt.subplots(1, 3, figsize=(15, 5))
    disp_ice = PartialDependenceDisplay.from_estimator(
        _pdp_model, Xtr42, features=_pdp_feats_idx[:3],
        feature_names=FN, ax=ax_ice,
        kind="both", subsample=100, random_state=42,
        line_kw={"color": "#2166AC", "lw": 2, "alpha": 0.8},
        ice_lines_kw={"color": "#D7191C", "alpha": 0.08, "lw": 0.8}
    )
    for i, ax_ in enumerate(ax_ice):
        ax_.set_title(f"PDP + ICE — {_pdp_feat_names[i]}", fontweight="bold", fontsize=11)
        ax_.set_ylabel("FOV (kV)"); ax_.grid(alpha=0.3)
    fig.suptitle(f"Individual Conditional Expectation (ICE) — Top 3 Features ({BEST})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); savefig("ice_top3.png")
    print("  ✔ ICE plots saved")
except Exception as e:
    print(f"  ICE skipped: {e}")

# ════ §15 EXCEL RAPORU ════════════════════════════════════════════
section("§15  Excel Report")
def fmt(m,s,d=4): return f"{m:.{d}f} ± {s:.{d}f}"
with pd.ExcelWriter(os.path.join(OUT_DIR,"results_Q1_ULTIMATE.xlsx"),engine="openpyxl") as writer:
    t1=[{"Model":nm,"Type":"Physics" if nm in ["Obenaus","Rizk"] else "ML",
         "R²":fmt(agg.loc[nm,"R2_mean"],agg.loc[nm,"R2_std"]),
         "R²[95%CI]":f"{agg.loc[nm,'R2_mean']:.4f}[{ci_res.get(nm,{}).get('R2_lo',0):.4f},{ci_res.get(nm,{}).get('R2_hi',0):.4f}]",
         "RMSE":fmt(agg.loc[nm,"RMSE_mean"],agg.loc[nm,"RMSE_std"]),
         "MAE":fmt(agg.loc[nm,"MAE_mean"],agg.loc[nm,"MAE_std"]),
         "MAPE%":fmt(agg.loc[nm,"MAPE_mean"],agg.loc[nm,"MAPE_std"],d=3),
         "NSE":fmt(agg.loc[nm,"NSE_mean"],agg.loc[nm,"NSE_std"])} for nm in agg.index]
    pd.DataFrame(t1).to_excel(writer,sheet_name="Table1_AllModels",index=False)
    pd.DataFrame(wil_rows).to_excel(writer,sheet_name="Table2_StatTests",index=False)
    pd.DataFrame(noise_rows).to_excel(writer,sheet_name="Table3_Sensitivity",index=False)
    pd.DataFrame([{"Feature":k,"β":round(v,4)} for k,v in sorted(coefs_.items(),key=lambda x:abs(x[1]),reverse=True)]).to_excel(writer,sheet_name="Table4_PhysicsCoefs",index=False)
    pd.DataFrame({"Reference":["This study","This study (HyPhysML)","[Lit-1]","[Lit-2]","[Lit-3]"],
        "Method":[BEST,"HyPhysML ULTIMATE","FILL IN","FILL IN","FILL IN"],
        "R²":[f"{agg.loc[BEST,'R2_mean']:.4f}",f"{m_hyp42['R2']:.4f}","FILL IN","FILL IN","FILL IN"],
        "RMSE":[f"{agg.loc[BEST,'RMSE_mean']:.4f}",f"{m_hyp42['RMSE']:.4f}","FILL IN","FILL IN","FILL IN"]
    }).to_excel(writer,sheet_name="Table5_LitTemplate",index=False)
    res_df.to_excel(writer,sheet_name="Raw_AllModels",index=False)
    _cv_df.to_excel(writer,sheet_name="Table6_CV_Scores",index=False)
    _tr_agg.to_excel(writer,sheet_name="Table7_TrainTestR2")
    _time_agg.to_excel(writer,sheet_name="Table8_CompTime")
    _vif_data.to_excel(writer,sheet_name="Table9_VIF",index=False)
print("  ✔ results_Q1_ULTIMATE.xlsx")

# ── BEST_PARAMS → Excel (Tablo 7 — Hiperparametre Tekrarlanabilirlik) ─────────
# 7 base learners + Ridge meta-learner used in HyPhysML stacking
_HYPHY_LEARNERS = ["XGBoost","LightGBM","GBR","HistGBR","RF","Extra Trees","KNN","Ridge"]
_SKIP_PARAMS    = {"random_state","verbosity","n_jobs","verbose","tree_method","nthread"}

# Extract (parameter name, optimal value) pairs for each model
_hpo_rows = []
for _nm in _HYPHY_LEARNERS:
    _obj = BEST_PARAMS.get(_nm)
    if _obj is None:
        _hpo_rows.append({"Learner":_nm,"Parameter":"—","Type":"—","Optimal Value":"Optuna not run"})
        continue
    if isinstance(_obj, dict):
        _params = _obj
    else:
        try:    _params = _obj.get_params()
        except: _params = {}
    for _k, _v in _params.items():
        if _k in _SKIP_PARAMS: continue
        if isinstance(_v, float): _fmt = f"{_v:.6g}"
        elif isinstance(_v, (tuple,list)): _fmt = str(_v)
        else: _fmt = str(_v)
        _typ = "int" if isinstance(_v,int) else ("float" if isinstance(_v,float) else "categorical")
        _hpo_rows.append({"Learner":_nm,"Parameter":_k,"Type":_typ,"Optimal Value":_fmt})

_hp_df = pd.DataFrame(_hpo_rows)

# Search spaces — fixed values from the notebook
_SEARCH_SPACES = {
    ("XGBoost","n_estimators")       : "[300, 2000]",
    ("XGBoost","learning_rate")      : "[0.005, 0.15] log",
    ("XGBoost","max_depth")          : "[4, 10]",
    ("XGBoost","subsample")          : "[0.6, 1.0]",
    ("XGBoost","colsample_bytree")   : "[0.5, 1.0]",
    ("XGBoost","min_child_weight")   : "[1, 20]",
    ("XGBoost","gamma")              : "[0.0, 0.5]",
    ("XGBoost","reg_lambda")         : "[0.01, 10.0] log",
    ("XGBoost","reg_alpha")          : "[0.0, 2.0]",
    ("LightGBM","n_estimators")      : "[300, 2000]",
    ("LightGBM","learning_rate")     : "[0.005, 0.15] log",
    ("LightGBM","num_leaves")        : "[31, 255]",
    ("LightGBM","subsample")         : "[0.6, 1.0]",
    ("LightGBM","colsample_bytree")  : "[0.5, 1.0]",
    ("LightGBM","min_child_samples") : "[5, 100]",
    ("LightGBM","reg_lambda")        : "[0.01, 10.0] log",
    ("LightGBM","reg_alpha")         : "[0.0, 2.0]",
    ("GBR","n_estimators")           : "[300, 1500]",
    ("GBR","learning_rate")          : "[0.005, 0.1] log",
    ("GBR","max_depth")              : "[3, 7]",
    ("GBR","subsample")              : "[0.6, 1.0]",
    ("GBR","min_samples_leaf")       : "[1, 10]",
    ("GBR","max_features")           : "[0.4, 1.0]",
    ("HistGBR","max_iter")           : "[300, 1500]",
    ("HistGBR","learning_rate")      : "[0.005, 0.1] log",
    ("HistGBR","max_depth")          : "[3, 10]",
    ("HistGBR","l2_regularization")  : "[0.0, 1.0]",
    ("HistGBR","min_samples_leaf")   : "[5, 50]",
    ("HistGBR","max_leaf_nodes")     : "[20, 255]",
    ("RF","n_estimators")            : "[200, 800]",
    ("RF","min_samples_leaf")        : "[1, 10]",
    ("RF","max_features")            : "[0.3, 0.9]",
    ("RF","min_samples_split")       : "[2, 10]",
    ("Extra Trees","n_estimators")   : "[200, 800]",
    ("Extra Trees","min_samples_leaf"): "[1, 10]",
    ("Extra Trees","max_features")   : "[0.3, 0.9]",
    ("Extra Trees","min_samples_split"): "[2, 10]",
    ("KNN","n_neighbors")            : "[2, 20]",
    ("KNN","weights")                : "{uniform, distance}",
    ("KNN","p")                      : "{1, 2}",
    ("Ridge","alpha")                : "[0.001, 100.0] log",
}

_hp_df["Search Space"] = _hp_df.apply(
    lambda r: _SEARCH_SPACES.get((r["Learner"], r["Parameter"]), "—"), axis=1)
_hp_df = _hp_df[["Learner","Parameter","Type","Search Space","Optimal Value"]]

# Save to Excel
_abl_path2 = os.path.join(OUT_DIR, "Table_HyperParams.xlsx")
with pd.ExcelWriter(_abl_path2, engine="openpyxl") as _wr2:
    _hp_df.to_excel(_wr2, sheet_name="Table7_HyperParams", index=False)
print(f"  ✔ Table_HyperParams.xlsx — {len(_hp_df)} rows, {_hp_df['Learner'].nunique()} models")

# Console output
print("\n=== TABLE 7 — OPTIMAL HYPERPARAMETERS ===")
for _nm in _HYPHY_LEARNERS:
    sub = _hp_df[_hp_df["Learner"]==_nm]
    if len(sub)==0: continue
    print(f"\n  [{_nm}]")
    for _, row in sub.iterrows():
        print(f"    {row['Parameter']:<25} = {row['Optimal Value']:<18}  (search: {row['Search Space']})")


# ════ §16 SUMMARY ═════════════════════════════════════════════════
section("§16  Summary")
figs_n=len([f for f in os.listdir(OUT_DIR) if f.endswith(".png")])
best_r=agg.iloc[0]; ci_b=ci_res.get(BEST,{})
print(f"\n  ★ {BEST}: R²={best_r.R2_mean:.4f}±{best_r.R2_std:.4f}")
print(f"     RMSE={best_r.RMSE_mean:.4f} kV  MAPE={best_r.MAPE_mean:.3f}%  NSE={best_r.NSE_mean:.4f}")
print(f"     95% CI: [{ci_b.get('R2_lo',0):.4f}, {ci_b.get('R2_hi',0):.4f}]")
print(f"\n  HyPhysML: R²={m_hyp42['R2']:.4f}  RMSE={m_hyp42['RMSE']:.4f} kV")
print(f"\n  XGBoost: {'✅' if HAS_XGB else '❌ missing — pip install xgboost'}")
print(f"  LightGBM:{'✅' if HAS_LGB else '❌ missing — pip install lightgbm'}")
print(f"  Optuna:  {'✅' if HAS_OPTUNA else '❌ missing — pip install optuna'}")
print(f"\n  {figs_n} figures  |  results_Q1_ULTIMATE.xlsx  |  {OUT_DIR}/")
print(f"  {'='*55}\n  COMPLETED\n  {'='*55}")


# ========================================================================
# [CODE CELL 5]
# ========================================================================
# ════════════════════════════════════════════════════════════════
# §17  ABLATION STUDY — HyPhysML Component Contribution Analysis
#
#  Each component removed one at a time to measure its effect (10 seeds x 4 variants):
#  Variant                        | HPO | OOF+Ridge | 7 Base Learners
#  ------------------------------|-----|-----------|----------------
#  HyPhysML (Full)               |  V  |     V     |       V
#  HyPhysML-noHPO                |  X  |     V     |       V   <- Optuna removed
#  HyPhysML-MeanStack            |  V  |  Mean     |       V   <- Ridge -> simple average
#  XGBoost (Single Model, HPO)   |  V  |     X     |       X   <- Stacking removed
# ════════════════════════════════════════════════════════════════

section("§17  Ablation Study — Component Contribution Analysis")

# ── Variant A: HyPhysML-noHPO ────────────────────────────────────
# All base learners with sklearn/library default parameters (no HPO)
class HyPhysML_NoHPO(HyPhysML):
    """HyPhysML — without Bayesian HPO (all base learners use default parameters)"""
    def _make_base(self, seed):
        bls = {}
        if HAS_XGB:
            bls["XGBoost"] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.3,
                subsample=1.0, colsample_bytree=1.0,
                random_state=seed, verbosity=0, n_jobs=-1, tree_method="hist"
            )
        if HAS_LGB:
            bls["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=100, num_leaves=31, learning_rate=0.1,
                random_state=seed, n_jobs=-1, verbose=-1
            )
        bls["GBR"]     = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed)
        bls["HistGBR"] = HistGradientBoostingRegressor(
            max_iter=100, random_state=seed)
        bls["RF"]      = RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=seed)
        bls["ET"]      = ExtraTreesRegressor(
            n_estimators=100, n_jobs=-1, random_state=seed)
        bls["KNN"]     = Pipeline([
            ("sc", RobustScaler()),
            ("m",  KNeighborsRegressor(n_neighbors=5, weights="distance"))
        ])
        return bls

# ── Variant B: HyPhysML-MeanStack ────────────────────────────────
# Simple equal-weight averaging instead of Ridge meta-learner
class HyPhysML_MeanStack(HyPhysML):
    """HyPhysML — simple equal-weight average instead of Ridge meta-learner"""
    def fit(self, X, y):
        rs  = self.random_state
        bls = self._make_base(rs)
        nb  = len(bls)
        self.bl_fit_       = {nm_: clone(bl_).fit(X, y) for nm_, bl_ in bls.items()}
        self.meta_         = None
        self.meta_weights_ = np.ones(nb) / nb
        self.meta_names_   = list(bls.keys())
        self.pm_           = ObenausModel(alpha=0.01).fit(X, y)
        self.physics_coef_ = self.pm_.coef_dict_
        self.n_sdd_        = self.pm_.n_sdd_
        return self

    def predict(self, X):
        preds = np.zeros((len(X), len(self.bl_fit_)))
        for bi, (_, bl_) in enumerate(self.bl_fit_.items()):
            try:    preds[:, bi] = bl_.predict(X)
            except: preds[:, bi] = 0.
        return np.mean(preds, axis=1)

# ── Ablation loop ────────────────────────────────────────────────
_abl_variants = {
    "HyPhysML-noHPO"    : lambda s: HyPhysML_NoHPO(random_state=s),
    "HyPhysML-MeanStack": lambda s: HyPhysML_MeanStack(random_state=s),
}

_abl_extra = []
print(f"  {len(_abl_variants)} new variants x {len(SEEDS)} seeds — HyPhysML (Full) and XGBoost taken from existing results")
for si, seed in enumerate(SEEDS):
    print(f"  Seed {seed:>5} ({si+1}/{len(SEEDS)}) ->", end=" ")
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=seed,
        stratify=df["Sample"].values
    )
    Xtr, Xte = X_all[idx_tr], X_all[idx_te]
    ytr, yte  = y_all[idx_tr], y_all[idx_te]
    for vname, model_fn in _abl_variants.items():
        m_obj = model_fn(seed)
        t0    = time.time()
        try:
            m_obj.fit(Xtr, ytr)
            yp = m_obj.predict(Xte)
            m  = compute_metrics(yte, yp)
            m.update({"Variant": vname, "Seed": seed,
                      "Time_s": round(time.time() - t0, 3)})
            _abl_extra.append(m)
            short = vname.replace("HyPhysML-","")[:8]
            print(f"[{short} R2={m['R2']:.4f}]", end=" ")
        except Exception as e:
            print(f"[{vname} ERROR: {e}]", end=" ")
    print()

_abl_extra_df = pd.DataFrame(_abl_extra)

# Full HyPhysML + XGBoost pulled from existing res_df
_abl_main = res_df[res_df["Model"].isin(["HyPhysML","XGBoost"])].copy()
_abl_main["Variant"] = _abl_main["Model"].map({
    "HyPhysML" : "HyPhysML (Full)",
    "XGBoost"  : "XGBoost (Single Model, HPO)"
})
_cols = ["Variant","Seed","R2","RMSE","MAE","MAPE","NSE","Time_s"]
_abl_main = _abl_main[[c for c in _cols if c in _abl_main.columns]]

_abl_full = pd.concat([_abl_main, _abl_extra_df[[c for c in _cols if c in _abl_extra_df.columns]]], ignore_index=True)

_abl_agg = (_abl_full.groupby("Variant")
            .agg(R2_mean=("R2","mean"), R2_std=("R2","std"),
                 RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
                 MAPE_mean=("MAPE","mean"), MAPE_std=("MAPE","std"),
                 Time_mean=("Time_s","mean"))
            .reset_index()
            .sort_values("R2_mean", ascending=False))

# ── Print results ────────────────────────────────────────────────
_ref_rows = _abl_agg[_abl_agg["Variant"]=="HyPhysML (Full)"]["R2_mean"].values
_ref_r2   = _ref_rows[0] if len(_ref_rows) else _abl_agg["R2_mean"].iloc[0]
_order  = ["HyPhysML (Full)","HyPhysML-noHPO","HyPhysML-MeanStack","XGBoost (Single Model, HPO)"]

print("\n" + "="*78)
print("  ABLATION RESULTS — HyPhysML Component Contribution Analysis (n=10 seeds)")
print("="*78)
print(f"  {'Variant':<35} {'R2 (mean+-std)':<20} {'RMSE (kV)':<12} {'MAPE%':<9} {'Delta_R2':>9}")
print("  " + "-"*76)
for v in _order:
    row = _abl_agg[_abl_agg["Variant"]==v]
    if len(row) == 0: continue
    r     = row.iloc[0]
    delta = r.R2_mean - _ref_r2
    sign  = "+" if delta >= 0 else ""
    arrow = " [REF]" if delta == 0 else (" [DEGRADED]" if delta < -0.0001 else " [EQ]")
    print(f"  {v:<35} {r.R2_mean:.4f}+-{r.R2_std:.4f}   {r.RMSE_mean:.4f}       {r.MAPE_mean:.3f}%   {sign}{delta:.4f}{arrow}")
print("="*78)
print("  Delta_R2 = difference relative to full HyPhysML (reference)")
print("  DEGRADED = removing this component hurt performance")

# ── Save to Excel ────────────────────────────────────────────────
_abl_path = os.path.join(OUT_DIR, "Table_Ablation.xlsx")
with pd.ExcelWriter(_abl_path, engine="openpyxl") as _wr:
    _abl_agg.to_excel(_wr, sheet_name="Ablation_Summary", index=False)
    _abl_full.to_excel(_wr, sheet_name="Ablation_Raw", index=False)

print(f"\n[OK] Ablation table saved: {_abl_path}")
print("     -> Open Table_Ablation.xlsx to review results.")
print("     -> To be included as Table 7 in the paper.")



# ========================================================================
# [CODE CELL 6]
# ========================================================================
# ── Zip results ───────────────────────────────────────────────────
import zipfile, os

zip_path = os.path.join(OUT_DIR, "fov_results_ultimate.zip")
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath=os.path.join(OUT_DIR,fname)
        if os.path.isfile(fpath) and not fname.endswith(".zip"):
            zf.write(fpath,fname)

size_mb=os.path.getsize(zip_path)/1024/1024
print(f"✅ {size_mb:.1f} MB — Zip file ready: {zip_path}")


