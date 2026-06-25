"""
=============================================================
  Credit Card Fraud Detection — Complete ML Pipeline
=============================================================
Learning outcomes covered:
  ✔ Synthetic dataset (mirrors Kaggle creditcard.csv structure, ~1% fraud)
  ✔ EDA & class imbalance visualisation
  ✔ Feature engineering (6 engineered features)
  ✔ Imbalance: SMOTE, RandomOverSampler, RandomUnderSampler
  ✔ Models: Logistic Regression, Random Forest, HistGradientBoosting
  ✔ Stratified K-Fold CV (k=5, SMOTE inside each fold → no leakage)
  ✔ AUC-PR vs AUC-ROC — why PR wins for imbalanced fraud data
  ✔ Threshold tuning (F1-optimal & cost-sensitive)
  ✔ Cost-sensitive evaluation (FN penalty × 10 vs FP)
  ✔ Feature importance
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

RNG = 42
np.random.seed(RNG)
OUT = "/mnt/user-data/outputs/"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SYNTHETIC DATASET
#   Mirrors the Kaggle creditcard.csv:
#     V1–V28  : PCA-transformed features (class-separated but noisy)
#     Time    : seconds since first transaction in 48-hour window
#     Amount  : transaction value (log-normal)
#     Class   : 0 = legit, 1 = fraud  (~1 % fraud)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1 — Generating Synthetic Credit-Card Dataset")
print("=" * 60)

N_LEGIT, N_FRAUD, nc = 8_000, 80, 28

# Mild mean shift on first 8 PCA components; high noise → realistic overlap
fraud_means = np.zeros(nc)
fraud_means[:8] = [-1.8, 1.2, -2.1, 0.9, -1.0, 1.4, -0.7, 0.8]

cov_legit = np.eye(nc) * 1.2
cov_fraud = np.eye(nc) * 2.0   # fraud is noisier / less predictable

V_legit = np.random.multivariate_normal(np.zeros(nc), cov_legit, N_LEGIT)
V_fraud  = np.random.multivariate_normal(fraud_means,  cov_fraud,  N_FRAUD)

cols_V = [f"V{i}" for i in range(1, 29)]

df_legit           = pd.DataFrame(V_legit, columns=cols_V)
df_legit["Time"]   = np.sort(np.random.uniform(0, 172_800, N_LEGIT))
df_legit["Amount"] = np.abs(np.random.lognormal(3.0, 1.5, N_LEGIT))
df_legit["Class"]  = 0

df_fraud           = pd.DataFrame(V_fraud, columns=cols_V)
df_fraud["Time"]   = np.random.uniform(0, 172_800, N_FRAUD)
df_fraud["Amount"] = np.abs(np.random.lognormal(4.5, 1.2, N_FRAUD))
df_fraud["Class"]  = 1

df = (pd.concat([df_legit, df_fraud], ignore_index=True)
        .sample(frac=1, random_state=RNG)
        .reset_index(drop=True))

print(f"  Rows       : {len(df):,}  ({N_LEGIT:,} legit + {N_FRAUD} fraud)")
print(f"  Fraud rate : {df['Class'].mean()*100:.2f}%")
print(f"  Missing    : {df.isna().sum().sum()}")
print(f"  Amount μ   : legit=${df[df.Class==0]['Amount'].mean():.1f}  "
      f"fraud=${df[df.Class==1]['Amount'].mean():.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2 — Exploratory Data Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EDA — Credit Card Fraud Dataset", fontsize=15,
             fontweight="bold", y=1.01)

# ── 2a: class imbalance ──────────────────────────────────────────────────────
counts = df["Class"].value_counts()
axes[0, 0].bar(["Legitimate (0)", "Fraud (1)"], counts.values,
               color=["#1976D2", "#D32F2F"], edgecolor="white", width=0.45)
axes[0, 0].set_title("Class Distribution  ⚠ Severe Imbalance", fontweight="bold")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0, 0].text(i, v + 25, f"{v:,}  ({v/len(df)*100:.1f}%)",
                    ha="center", fontsize=10, fontweight="bold")
axes[0, 0].set_ylim(0, N_LEGIT * 1.18)

# ── 2b: amount distribution ──────────────────────────────────────────────────
axes[0, 1].hist(df[df.Class==0]["Amount"], bins=60, alpha=0.6,
                label="Legitimate", color="#1976D2", density=True)
axes[0, 1].hist(df[df.Class==1]["Amount"], bins=30, alpha=0.85,
                label="Fraud", color="#D32F2F", density=True)
axes[0, 1].set_title("Transaction Amount Distribution", fontweight="bold")
axes[0, 1].set_xlabel("Amount ($)"); axes[0, 1].set_ylabel("Density")
axes[0, 1].set_xlim(0, 3000); axes[0, 1].legend()

# ── 2c: V1 distribution (top discriminative PCA component) ──────────────────
axes[1, 0].hist(df[df.Class==0]["V1"], bins=60, alpha=0.6,
                label="Legitimate", color="#1976D2", density=True)
axes[1, 0].hist(df[df.Class==1]["V1"], bins=30, alpha=0.85,
                label="Fraud", color="#D32F2F", density=True)
axes[1, 0].set_title("V1 — Top Discriminative PCA Component", fontweight="bold")
axes[1, 0].set_xlabel("V1 Value"); axes[1, 0].set_ylabel("Density")
axes[1, 0].legend()

# ── 2d: correlation heatmap ──────────────────────────────────────────────────
top10 = (df.corr()["Class"].drop("Class").abs()
           .sort_values(ascending=False).head(10).index.tolist())
sns.heatmap(df[top10 + ["Class"]].corr(), ax=axes[1, 1],
            cmap="RdBu_r", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, linewidths=0.4)
axes[1, 1].set_title("Top-10 Features × Class Correlation", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT + "01_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → 01_eda.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3 — Feature Engineering")
print("=" * 60)

df["Log_Amount"]      = np.log1p(df["Amount"])
df["Hour"]            = (df["Time"] // 3600) % 24
df["Is_Night"]        = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)
df["V1_V3_interact"]  = df["V1"] * df["V3"]
df["V4_V14_interact"] = df["V4"] * df["V14"]
df["High_Amount"]     = (df["Amount"] > df["Amount"].median()).astype(int)

new_feats = ["Log_Amount", "Hour", "Is_Night",
             "V1_V3_interact", "V4_V14_interact", "High_Amount"]
print(f"  Added {len(new_feats)} features: {new_feats}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
DROP  = ["Class", "Time", "Amount"]
FEATS = [c for c in df.columns if c not in DROP]

X = df[FEATS].values
y = df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RNG)

print(f"\n  Train : {X_train.shape}  fraud={Counter(y_train)[1]}")
print(f"  Test  : {X_test.shape}   fraud={Counter(y_test)[1]}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — IMBALANCE HANDLING STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4 — Resampling Strategies")
print("=" * 60)

sc        = StandardScaler()
X_tr_sc   = sc.fit_transform(X_train)
X_te_sc   = sc.transform(X_test)

X_sm,  y_sm  = SMOTE(random_state=RNG).fit_resample(X_tr_sc, y_train)
X_ros, y_ros = RandomOverSampler(random_state=RNG).fit_resample(X_tr_sc, y_train)
X_rus, y_rus = RandomUnderSampler(random_state=RNG).fit_resample(X_tr_sc, y_train)

for name, yr in [("SMOTE (Synthetic)",  y_sm),
                 ("Random Over-Sample",  y_ros),
                 ("Random Under-Sample", y_rus)]:
    c = Counter(yr)
    print(f"  {name:24s} → class 0: {c[0]:5d}  class 1: {c[1]:5d}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — STRATIFIED K-FOLD CROSS-VALIDATION
#   SMOTE runs inside each fold → zero data leakage
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5 — 5-Fold Stratified CV (SMOTE inside each fold)")
print("=" * 60)

models = {
    "Logistic Regression":   LogisticRegression(
        class_weight="balanced", max_iter=500, C=0.05, random_state=RNG),
    "Random Forest":         RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced",
        random_state=RNG, n_jobs=1),
    "Hist Grad Boosting":    HistGradientBoostingClassifier(
        max_iter=100, max_depth=4, random_state=RNG),
}

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
cv_res = {}

for name, clf in models.items():
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=RNG)),
        ("clf",    clf),
    ])
    pr_sc  = cross_val_score(pipe, X_train, y_train, cv=cv,
                              scoring="average_precision")
    roc_sc = cross_val_score(pipe, X_train, y_train, cv=cv,
                              scoring="roc_auc")
    cv_res[name] = {"AUC-PR": pr_sc, "AUC-ROC": roc_sc}
    print(f"  {name:24s}  AUC-PR={pr_sc.mean():.4f}±{pr_sc.std():.4f}"
          f"  AUC-ROC={roc_sc.mean():.4f}±{roc_sc.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FINAL MODEL TRAINING  (on SMOTE-balanced data)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 6 — Training Final Models on SMOTE Data")
print("=" * 60)

fitted = {}
for name, clf in models.items():
    clf.fit(X_sm, y_sm)
    fitted[name] = clf
    print(f"  Trained: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — EVALUATION: AUC-ROC vs AUC-PR
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 7 — Test-Set Evaluation")
print("=" * 60)

COLOURS = ["#1976D2", "#FF6F00", "#2E7D32"]
res     = {}

for name, clf in fitted.items():
    y_prob = clf.predict_proba(X_te_sc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    fpr,  tpr,  _  = roc_curve(y_test, y_prob)
    prec, rec_c, _ = precision_recall_curve(y_test, y_prob)
    res[name] = dict(
        y_prob=y_prob, y_pred=y_pred,
        auc_roc=roc_auc_score(y_test, y_prob),
        auc_pr =average_precision_score(y_test, y_prob),
        f1     =f1_score(y_test, y_pred, zero_division=0),
        prec   =precision_score(y_test, y_pred, zero_division=0),
        rec    =recall_score(y_test, y_pred, zero_division=0),
        fpr=fpr, tpr=tpr, prec_c=prec, rec_c=rec_c,
    )
    print(f"  {name:24s}  AUC-ROC={res[name]['auc_roc']:.4f}"
          f"  AUC-PR={res[name]['auc_pr']:.4f}"
          f"  F1={res[name]['f1']:.4f}"
          f"  Prec={res[name]['prec']:.4f}"
          f"  Rec={res[name]['rec']:.4f}")

# ─── ROC + PR curves ─────────────────────────────────────────────────────────
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("AUC-ROC vs Precision-Recall Curves — All Models",
             fontsize=14, fontweight="bold")

for (name, r), col in zip(res.items(), COLOURS):
    ax_roc.plot(r["fpr"], r["tpr"], color=col, lw=2.2,
                label=f"{name}  (AUC={r['auc_roc']:.3f})")
    ax_pr.plot(r["rec_c"], r["prec_c"], color=col, lw=2.2,
               label=f"{name}  (AP={r['auc_pr']:.3f})")

ax_roc.plot([0,1],[0,1],"k--",lw=1,label="Random baseline")
ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve\n⚠  High AUC-ROC even on random classifier "
                  "with imbalanced data", fontsize=10)
ax_roc.legend(fontsize=9); ax_roc.grid(alpha=0.3)

baseline_pr = y.mean()
ax_pr.axhline(y=baseline_pr, color="k", ls="--", lw=1,
              label=f"Random baseline (prevalence={baseline_pr:.3f})")
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve\n✔  Correct metric when positives "
                "are rare — lower baseline makes gains visible", fontsize=10)
ax_pr.legend(fontsize=9); ax_pr.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT + "02_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → 02_roc_pr_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — THRESHOLD TUNING
#   Default = 0.5.  In fraud detection we usually lower it to boost recall.
#   We find two optimal thresholds:
#     (a) maximises F1-score
#     (b) minimises cost  where  FN costs 10× more than FP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 8 — Threshold Tuning (best model by AUC-PR)")
print("=" * 60)

best_name = max(res, key=lambda n: res[n]["auc_pr"])
y_prob    = res[best_name]["y_prob"]
FN_COST, FP_COST = 10, 1

thresholds  = np.linspace(0.01, 0.99, 300)
f1_arr, cost_arr, prec_arr, rec_arr = [], [], [], []

for thr in thresholds:
    yt = (y_prob >= thr).astype(int)
    f1_arr.append(f1_score(y_test, yt, zero_division=0))
    prec_arr.append(precision_score(y_test, yt, zero_division=0))
    rec_arr.append(recall_score(y_test, yt, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, yt, labels=[0,1]).ravel()
    cost_arr.append(fn * FN_COST + fp * FP_COST)

best_f1_thr   = thresholds[np.argmax(f1_arr)]
best_cost_thr = thresholds[np.argmin(cost_arr)]

print(f"  Best model     : {best_name}")
print(f"  Default (0.50) : F1={f1_score(y_test,(y_prob>=0.5).astype(int),zero_division=0):.4f}")
print(f"  Best-F1  thr   : {best_f1_thr:.3f}  →  F1={max(f1_arr):.4f}")
print(f"  Min-cost thr   : {best_cost_thr:.3f}  →  Cost={min(cost_arr)}")

# ─── Threshold tuning plots ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Threshold Tuning — {best_name}", fontsize=13, fontweight="bold")

# Prec / Rec / F1
axes[0].plot(thresholds, prec_arr, "#1976D2", lw=2, label="Precision")
axes[0].plot(thresholds, rec_arr,  "#D32F2F", lw=2, label="Recall")
axes[0].plot(thresholds, f1_arr,   "#2E7D32", lw=2, label="F1")
axes[0].axvline(best_f1_thr, color="k", ls="--", lw=1.5,
                label=f"Best-F1  thr={best_f1_thr:.2f}")
axes[0].axvline(0.5, color="grey", ls=":", lw=1, label="Default=0.50")
axes[0].set_xlabel("Decision Threshold"); axes[0].set_ylabel("Score")
axes[0].set_title("Precision / Recall / F1 vs Threshold")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# F1
axes[1].plot(thresholds, f1_arr, "#2E7D32", lw=2.2)
axes[1].fill_between(thresholds, f1_arr, alpha=0.12, color="#2E7D32")
axes[1].axvline(best_f1_thr, color="red", ls="--", lw=1.5,
                label=f"Opt thr={best_f1_thr:.3f}\nF1={max(f1_arr):.4f}")
axes[1].axvline(0.5, color="grey", ls=":", lw=1, label="Default=0.50")
axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("F1 Score")
axes[1].set_title("F1-Optimal Threshold")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

# Cost
axes[2].plot(thresholds, cost_arr, "#D32F2F", lw=2.2)
axes[2].fill_between(thresholds, cost_arr, alpha=0.08, color="#D32F2F")
axes[2].axvline(best_cost_thr, color="navy", ls="--", lw=1.5,
                label=f"Opt thr={best_cost_thr:.3f}\nCost={min(cost_arr)}")
axes[2].axvline(0.5, color="grey", ls=":", lw=1, label="Default=0.50")
axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("Total Cost")
axes[2].set_title(f"Cost-Sensitive  (FN×{FN_COST} + FP×{FP_COST})")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT + "03_threshold_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → 03_threshold_tuning.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — COST-SENSITIVE EVALUATION AT THREE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 9 — Cost-Sensitive Metrics at Key Thresholds")
print("=" * 60)

for label, thr in [("Default (0.50)",           0.50),
                   (f"Best-F1  ({best_f1_thr:.2f})",  best_f1_thr),
                   (f"Min-Cost ({best_cost_thr:.2f})", best_cost_thr)]:
    yt = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yt, labels=[0,1]).ravel()
    tc = fn*FN_COST + fp*FP_COST
    print(f"\n  [{label}]")
    print(f"    TP={tp:3d}  FP={fp:4d}  FN={fn:3d}  TN={tn:5d}")
    print(f"    Precision={precision_score(y_test,yt,zero_division=0):.3f}  "
          f"Recall={recall_score(y_test,yt,zero_division=0):.3f}  "
          f"F1={f1_score(y_test,yt,zero_division=0):.3f}")
    print(f"    Cost = {fn}×{FN_COST}(missed fraud) + "
          f"{fp}×{FP_COST}(false alarm) = {tc}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — CONFUSION MATRICES  (at best-F1 threshold)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Confusion Matrices at Threshold = {best_f1_thr:.2f}",
             fontsize=13, fontweight="bold")

for ax, (name, r), col in zip(axes, res.items(), COLOURS):
    yt = (r["y_prob"] >= best_f1_thr).astype(int)
    cm = confusion_matrix(y_test, yt, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred Legit", "Pred Fraud"],
                yticklabels=["Actual Legit", "Actual Fraud"],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"{name}\nAP={r['auc_pr']:.3f} | FN={fn} missed | FP={fp} false alarm")

plt.tight_layout()
plt.savefig(OUT + "04_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → 04_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — CV COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("5-Fold Stratified CV  (SMOTE inside each fold — no leakage)",
             fontsize=13, fontweight="bold")

for ax, metric in zip(axes, ["AUC-PR", "AUC-ROC"]):
    means = [cv_res[m][metric].mean() for m in cv_res]
    stds  = [cv_res[m][metric].std()  for m in cv_res]
    bars  = ax.bar(list(cv_res), means, yerr=stds, capsize=7,
                   color=COLOURS, edgecolor="white", alpha=0.88)
    ax.set_title(f"CV {metric}  (mean ± std)", fontweight="bold")
    ax.set_ylim(0, 1.18); ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.03,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT + "05_cv_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → 05_cv_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 13 — FEATURE IMPORTANCE  (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════
rf_model    = fitted["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=FEATS)
importances = importances.sort_values(ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#FF6F00" if "interact" in f or "Log" in f or "Night" in f
          else "#1976D2" for f in importances.index]
bars = ax.barh(importances.index, importances.values,
               color=colors, edgecolor="white", alpha=0.88)
ax.set_title("Random Forest — Top-20 Feature Importances\n"
             "(Orange = engineered features)", fontsize=12, fontweight="bold")
ax.set_xlabel("Mean Decrease in Impurity")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.0003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)

from matplotlib.patches import Patch
legend_handles = [Patch(color="#1976D2", label="Original PCA features"),
                  Patch(color="#FF6F00", label="Engineered features")]
ax.legend(handles=legend_handles, fontsize=9)
plt.tight_layout()
plt.savefig(OUT + "06_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → 06_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 14 — IMBALANCE STRATEGY COMPARISON  (same RF, three resampling methods)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 10 — Resampling Strategy Comparison (Random Forest)")
print("=" * 60)

strat_data = {
    "SMOTE":             (X_sm,  y_sm),
    "RandomOverSample":  (X_ros, y_ros),
    "RandomUnderSample": (X_rus, y_rus),
}
strat_metrics = {}
rf_cmp = RandomForestClassifier(n_estimators=100, max_depth=8,
                                 random_state=RNG, n_jobs=1)

for sname, (Xr, yr) in strat_data.items():
    rf_cmp.fit(Xr, yr)
    yp = rf_cmp.predict_proba(X_te_sc)[:, 1]
    strat_metrics[sname] = {
        "AUC-PR":  average_precision_score(y_test, yp),
        "AUC-ROC": roc_auc_score(y_test, yp),
        "F1":      f1_score(y_test, (yp>=0.5).astype(int), zero_division=0),
    }
    print(f"  {sname:22s}  AUC-PR={strat_metrics[sname]['AUC-PR']:.4f}"
          f"  AUC-ROC={strat_metrics[sname]['AUC-ROC']:.4f}"
          f"  F1={strat_metrics[sname]['F1']:.4f}")

x, width = np.arange(len(strat_metrics)), 0.25
fig, ax = plt.subplots(figsize=(10, 5))
metric_colors = ["#1976D2", "#FF6F00", "#2E7D32"]
for i, (metric, col) in enumerate(zip(["AUC-PR", "AUC-ROC", "F1"], metric_colors)):
    vals = [strat_metrics[s][metric] for s in strat_metrics]
    bars = ax.bar(x + i*width, vals, width, label=metric, color=col, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x + width); ax.set_xticklabels(list(strat_metrics), rotation=5)
ax.set_ylim(0, 1.18); ax.set_ylabel("Score")
ax.set_title("Resampling Strategy Comparison — Random Forest",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "07_imbalance_strategy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → 07_imbalance_strategy_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 15 — FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FINAL MODEL SUMMARY")
print("=" * 60)

rows = []
for name, r in res.items():
    rows.append({
        "Model":         name,
        "CV AUC-PR":    round(cv_res[name]["AUC-PR"].mean(), 4),
        "CV AUC-ROC":   round(cv_res[name]["AUC-ROC"].mean(), 4),
        "Test AUC-PR":  round(r["auc_pr"], 4),
        "Test AUC-ROC": round(r["auc_roc"], 4),
        "Test F1":      round(r["f1"], 4),
        "Precision":    round(r["prec"], 4),
        "Recall":       round(r["rec"], 4),
    })

summary = pd.DataFrame(rows).set_index("Model")
print(summary.to_string())
summary.to_csv(OUT + "model_summary.csv")
print("\n  → model_summary.csv")

print("\n" + "=" * 60)
print("  ✅  All outputs saved to /mnt/user-data/outputs/")
print("=" * 60)
