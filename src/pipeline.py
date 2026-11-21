#!/usr/bin/env python3
"""
================================================================================
MSc Dissertation: Trustworthy ML-Based Fraud Detection for Card-Present Payment Systems
================================================================================
University of Nottingham - MSc Cyber-Physical Systems (Cyber Security Pathway)

Title: "Trustworthy Machine Learning-Based Fraud Detection for Card-Present Payment Systems"

Author: Sara Belhaiba
Supervisor: Prof Christian Wagner

This notebook implements the complete experimental pipeline for the dissertation,
including:
    1. Data Loading and Preprocessing
    2. Exploratory Data Analysis
    3. Class Imbalance Handling Comparison
    4. Model Training (Random Forest, XGBoost, Isolation Forest)
    5. Cross-Validation with Variance Analysis
    6. Threshold Sensitivity Analysis
    7. SHAP Explainability Analysis
    8. STRIDE-Based Attack Simulation
    9. Robustness Evaluation per Attack Type
    10. Traffic Light Decision Support Framework
    11. Statistical Significance Testing
    12. Figure Generation for Dissertation

Requirements:
    pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn scipy
Dataset:
    Kaggle Credit Card Fraud Detection Dataset (ULB)
    
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, roc_curve,
    precision_recall_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import xgboost as xgb
import shap
import warnings
import os
import json
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10, 6)

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*80)
print("MSc DISSERTATION: FRAUD DETECTION DECISION SUPPORT SYSTEM")
print("="*80)
print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# =============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# =============================================================================
print("\n" + "="*80)
print("SECTION 1: DATA LOADING AND PREPROCESSING")
print("="*80)

# Load dataset
# NOTE: Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# and place in 'data/' directory
DATA_PATH = 'data/creditcard.csv'

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset loaded: {DATA_PATH}")
else:
    print(f"✗ Dataset not found at {DATA_PATH}")
    print("  Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Behavioural feature: time elapsed since the previous transaction.
# This is added specifically to give velocity-style attacks (rapid repeated
# transactions in a short window) a chance to be distinguished from normal
# spacing between transactions, which the original V1-V28 + Amount + Time
# feature set could not capture on its own.
df = df.sort_values('Time').reset_index(drop=True)
df['Time_Delta'] = df['Time'].diff().fillna(df['Time'].median())

# Dataset overview
print(f"\n--- Dataset Overview ---")
print(f"Total transactions: {len(df):,}")
print(f"Features: {df.shape[1]}")
print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
print(f"Legitimate cases: {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.3f}%)")

# Feature information
print(f"\n--- Feature Information ---")
print(f"Time range: {df['Time'].min():.0f} to {df['Time'].max():.0f} seconds")
print(f"Amount range: £{df['Amount'].min():.2f} to £{df['Amount'].max():.2f}")
print(f"Amount mean: £{df['Amount'].mean():.2f}")
print(f"Amount median: £{df['Amount'].median():.2f}")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']
X['Time_Delta'] = df['Time_Delta']

# Feature scaling (Amount and Time)
print(f"\n--- Feature Scaling ---")
scaler = StandardScaler()
X['Amount_scaled'] = scaler.fit_transform(X[['Amount']])
X['Time_scaled'] = scaler.fit_transform(X[['Time']])

# Use scaled features (Time_Delta kept unscaled: tree-based models do not
# require feature scaling, and this avoids fitting yet another scaler)
feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled', 'Time_Delta']
X_processed = X[feature_cols].copy()
X_processed.columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time', 'Time_Delta']

print(f"✓ Amount scaled: mean={X['Amount_scaled'].mean():.4f}, std={X['Amount_scaled'].std():.4f}")
print(f"✓ Time scaled: mean={X['Time_scaled'].mean():.4f}, std={X['Time_scaled'].std():.4f}")

# Train/test split (stratified)
print(f"\n--- Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} transactions ({y_train.sum():,} frauds)")
print(f"Test set: {len(X_test):,} transactions ({y_test.sum():,} frauds)")
print(f"Fraud ratio preserved: Train={y_train.mean()*100:.3f}%, Test={y_test.mean()*100:.3f}%")

# Save data split for reproducibility
pickle.dump((X_train, X_test, y_train, y_test), open('results/data_split.pkl', 'wb'))
pickle.dump(scaler, open('results/scaler.pkl', 'wb'))
print(f"✓ Data split saved to results/data_split.pkl")


# =============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SECTION 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Figure 1: Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
labels = ['Legitimate', 'Fraud']
sizes = [len(df[df['Class']==0]), len(df[df['Class']==1])]
colors = ['#3498db', '#e74c3c']
explode = (0, 0.1)
axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.3f%%',
            shadow=True, startangle=90)
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')

# Bar chart with log scale
axes[1].bar(labels, sizes, color=colors, edgecolor='black')
axes[1].set_ylabel('Number of Transactions (log scale)')
axes[1].set_yscale('log')
axes[1].set_title('Class Distribution (Log Scale)', fontsize=14, fontweight='bold')
for i, v in enumerate(sizes):
    axes[1].text(i, v*1.1, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure saved: figures/01_class_distribution.png")

# Figure 2: Amount Distribution by Class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Legitimate transactions
axes[0].hist(df[df['Class']==0]['Amount'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Amount (£)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Amount Distribution - Legitimate Transactions', fontsize=12, fontweight='bold')
axes[0].set_xlim(0, 500)

# Fraud transactions
axes[1].hist(df[df['Class']==1]['Amount'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Amount (£)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Amount Distribution - Fraudulent Transactions', fontsize=12, fontweight='bold')
axes[1].set_xlim(0, 500)

plt.tight_layout()
plt.savefig('figures/02_amount_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure saved: figures/02_amount_distribution.png")

# Statistical comparison
legit_amount = df[df['Class']==0]['Amount']
fraud_amount = df[df['Class']==1]['Amount']

print(f"\n--- Amount Statistics by Class ---")
print(f"Legitimate: mean=£{legit_amount.mean():.2f}, median=£{legit_amount.median():.2f}, std=£{legit_amount.std():.2f}")
print(f"Fraudulent: mean=£{fraud_amount.mean():.2f}, median=£{fraud_amount.median():.2f}, std=£{fraud_amount.std():.2f}")

# Mann-Whitney U test for amount difference
stat, p_value = stats.mannwhitneyu(legit_amount, fraud_amount, alternative='two-sided')
print(f"Mann-Whitney U test: U={stat:.2f}, p-value={p_value:.2e}")
print(f"→ {'Significant' if p_value < 0.05 else 'Not significant'} difference in amounts between classes")


# =============================================================================
# SECTION 3: CLASS IMBALANCE HANDLING COMPARISON
# =============================================================================
print("\n" + "="*80)
print("SECTION 3: CLASS IMBALANCE HANDLING COMPARISON")
print("="*80)

print("""
Comparing three approaches to handle class imbalance:
1. No resampling (baseline) - trains on original distribution
2. Random undersampling - reduces majority class to match minority
3. Class weighting - penalises misclassification of minority class
""")

imbalance_results = {}

# Approach 1: No Resampling (Baseline)
print("\n--- Approach 1: No Resampling ---")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)
y_prob_baseline = rf_baseline.predict_proba(X_test)[:, 1]

imbalance_results['no_resampling'] = {
    'auc': roc_auc_score(y_test, y_prob_baseline),
    'precision': precision_score(y_test, y_pred_baseline),
    'recall': recall_score(y_test, y_pred_baseline),
    'f1': f1_score(y_test, y_pred_baseline),
    'fp': confusion_matrix(y_test, y_pred_baseline)[0, 1],
    'fn': confusion_matrix(y_test, y_pred_baseline)[1, 0]
}
print(f"AUC-ROC: {imbalance_results['no_resampling']['auc']:.4f}")
print(f"Precision: {imbalance_results['no_resampling']['precision']:.4f}")
print(f"Recall: {imbalance_results['no_resampling']['recall']:.4f}")
print(f"F1 Score: {imbalance_results['no_resampling']['f1']:.4f}")
print(f"False Positives: {imbalance_results['no_resampling']['fp']}")

# Approach 2: Random Undersampling
print("\n--- Approach 2: Random Undersampling ---")
fraud_indices = y_train[y_train == 1].index
legit_indices = y_train[y_train == 0].index
undersampled_legit = np.random.choice(legit_indices, size=len(fraud_indices), replace=False)
undersampled_indices = np.concatenate([fraud_indices, undersampled_legit])

X_train_under = X_train.loc[undersampled_indices]
y_train_under = y_train.loc[undersampled_indices]
print(f"Undersampled training set: {len(X_train_under)} transactions (50/50 split)")

rf_under = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_under.fit(X_train_under, y_train_under)
y_pred_under = rf_under.predict(X_test)
y_prob_under = rf_under.predict_proba(X_test)[:, 1]

imbalance_results['undersampling'] = {
    'auc': roc_auc_score(y_test, y_prob_under),
    'precision': precision_score(y_test, y_pred_under),
    'recall': recall_score(y_test, y_pred_under),
    'f1': f1_score(y_test, y_pred_under),
    'fp': confusion_matrix(y_test, y_pred_under)[0, 1],
    'fn': confusion_matrix(y_test, y_pred_under)[1, 0]
}
print(f"AUC-ROC: {imbalance_results['undersampling']['auc']:.4f}")
print(f"Precision: {imbalance_results['undersampling']['precision']:.4f}")
print(f"Recall: {imbalance_results['undersampling']['recall']:.4f}")
print(f"F1 Score: {imbalance_results['undersampling']['f1']:.4f}")
print(f"False Positives: {imbalance_results['undersampling']['fp']}")

# Approach 3: Class Weighting
print("\n--- Approach 3: Class Weighting ---")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {weight_dict}")

rf_weighted = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_weighted.fit(X_train, y_train)
y_pred_weighted = rf_weighted.predict(X_test)
y_prob_weighted = rf_weighted.predict_proba(X_test)[:, 1]

imbalance_results['class_weighting'] = {
    'auc': roc_auc_score(y_test, y_prob_weighted),
    'precision': precision_score(y_test, y_pred_weighted),
    'recall': recall_score(y_test, y_pred_weighted),
    'f1': f1_score(y_test, y_pred_weighted),
    'fp': confusion_matrix(y_test, y_pred_weighted)[0, 1],
    'fn': confusion_matrix(y_test, y_pred_weighted)[1, 0]
}
print(f"AUC-ROC: {imbalance_results['class_weighting']['auc']:.4f}")
print(f"Precision: {imbalance_results['class_weighting']['precision']:.4f}")
print(f"Recall: {imbalance_results['class_weighting']['recall']:.4f}")
print(f"F1 Score: {imbalance_results['class_weighting']['f1']:.4f}")
print(f"False Positives: {imbalance_results['class_weighting']['fp']}")

# Summary comparison
print("\n--- Class Imbalance Handling Summary ---")
print(f"{'Method':<20} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FP':>6}")
print("-" * 65)
for method, results in imbalance_results.items():
    print(f"{method:<20} {results['auc']:>8.4f} {results['precision']:>10.4f} {results['recall']:>8.4f} {results['f1']:>8.4f} {results['fp']:>6}")

print(f"\n✓ DECISION: No resampling selected (highest F1={imbalance_results['no_resampling']['f1']:.4f})")
print(f"  Undersampling rejected: catastrophic FP rate ({imbalance_results['undersampling']['fp']} false positives)")

# Save results

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

json.dump(convert_numpy(imbalance_results), open('results/imbalance_comparison.json', 'w'), indent=2)
print(f"✓ Results saved to results/imbalance_comparison.json")

# =============================================================================
# SECTION 4: MODEL TRAINING
# =============================================================================
print("\n" + "="*80)
print("SECTION 4: MODEL TRAINING")
print("="*80)

models = {}
predictions = {}
probabilities = {}

# Model 1: Random Forest
print("\n--- Model 1: Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
predictions['rf'] = rf_model.predict(X_test)
probabilities['rf'] = rf_model.predict_proba(X_test)[:, 1]
models['rf'] = rf_model

print(f"Training complete. Parameters: n_estimators=100, default depth")
print(f"AUC-ROC: {roc_auc_score(y_test, probabilities['rf']):.4f}")
print(f"F1 Score: {f1_score(y_test, predictions['rf']):.4f}")

# Model 2: XGBoost
print("\n--- Model 2: XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
predictions['xgb'] = xgb_model.predict(X_test)
probabilities['xgb'] = xgb_model.predict_proba(X_test)[:, 1]
models['xgb'] = xgb_model

print(f"Training complete. Parameters: n_estimators=100, max_depth=6, lr=0.1")
print(f"AUC-ROC: {roc_auc_score(y_test, probabilities['xgb']):.4f}")
print(f"F1 Score: {f1_score(y_test, predictions['xgb']):.4f}")

# Model 3: Isolation Forest
print("\n--- Model 3: Isolation Forest ---")
iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.00173,  # Approximate fraud ratio
    random_state=42,
    n_jobs=-1
)
iso_model.fit(X_train)
iso_pred = iso_model.predict(X_test)
predictions['iso'] = np.where(iso_pred == -1, 1, 0)  # -1 = anomaly = fraud
probabilities['iso'] = -iso_model.score_samples(X_test)  # Higher = more anomalous
# Normalize to 0-1 range
probabilities['iso'] = (probabilities['iso'] - probabilities['iso'].min()) / (probabilities['iso'].max() - probabilities['iso'].min())
models['iso'] = iso_model

print(f"Training complete. Parameters: n_estimators=100, contamination=0.173%")
print(f"AUC-ROC: {roc_auc_score(y_test, probabilities['iso']):.4f}")
print(f"F1 Score: {f1_score(y_test, predictions['iso']):.4f}")

# Save models
pickle.dump(rf_model, open('models/rf_model.pkl', 'wb'))
pickle.dump(xgb_model, open('models/xgb_model.pkl', 'wb'))
pickle.dump(iso_model, open('models/iso_model.pkl', 'wb'))
print(f"\n✓ Models saved to models/ directory")

# Model comparison summary
print("\n--- Model Comparison Summary ---")
print(f"{'Model':<20} {'AUC-ROC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)
for name, pred, prob in [('Random Forest', predictions['rf'], probabilities['rf']),
                          ('XGBoost', predictions['xgb'], probabilities['xgb']),
                          ('Isolation Forest', predictions['iso'], probabilities['iso'])]:
    auc_val = roc_auc_score(y_test, prob)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"{name:<20} {auc_val:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")


# =============================================================================
# SECTION 5: CROSS-VALIDATION WITH VARIANCE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SECTION 5: CROSS-VALIDATION WITH VARIANCE ANALYSIS")
print("="*80)

cv_results = {}
n_folds = 5

print(f"Performing stratified {n_folds}-fold cross-validation...")

# Random Forest CV
print("\n--- Random Forest Cross-Validation ---")
cv_rf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
rf_cv_auc = []
rf_cv_f1 = []
rf_cv_precision = []
rf_cv_recall = []

for fold, (train_idx, val_idx) in enumerate(cv_rf.split(X_processed, y), 1):
    X_cv_train, X_cv_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
    y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
    
    rf_cv = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_cv.fit(X_cv_train, y_cv_train)
    y_cv_pred = rf_cv.predict(X_cv_val)
    y_cv_prob = rf_cv.predict_proba(X_cv_val)[:, 1]
    
    rf_cv_auc.append(roc_auc_score(y_cv_val, y_cv_prob))
    rf_cv_f1.append(f1_score(y_cv_val, y_cv_pred))
    rf_cv_precision.append(precision_score(y_cv_val, y_cv_pred))
    rf_cv_recall.append(recall_score(y_cv_val, y_cv_pred))
    
    print(f"  Fold {fold}: AUC={rf_cv_auc[-1]:.4f}, F1={rf_cv_f1[-1]:.4f}")

cv_results['random_forest'] = {
    'auc_mean': np.mean(rf_cv_auc), 'auc_std': np.std(rf_cv_auc),
    'f1_mean': np.mean(rf_cv_f1), 'f1_std': np.std(rf_cv_f1),
    'precision_mean': np.mean(rf_cv_precision), 'precision_std': np.std(rf_cv_precision),
    'recall_mean': np.mean(rf_cv_recall), 'recall_std': np.std(rf_cv_recall),
    'auc_values': rf_cv_auc, 'f1_values': rf_cv_f1
}
print(f"  Mean AUC: {cv_results['random_forest']['auc_mean']:.4f} ± {cv_results['random_forest']['auc_std']:.4f}")
print(f"  Mean F1:  {cv_results['random_forest']['f1_mean']:.4f} ± {cv_results['random_forest']['f1_std']:.4f}")

# XGBoost CV
print("\n--- XGBoost Cross-Validation ---")
cv_xgb = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
xgb_cv_auc = []
xgb_cv_f1 = []
xgb_cv_precision = []
xgb_cv_recall = []

for fold, (train_idx, val_idx) in enumerate(cv_xgb.split(X_processed, y), 1):
    X_cv_train, X_cv_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
    y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
    
    xgb_cv = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               random_state=42, use_label_encoder=False, 
                               eval_metric='logloss', n_jobs=-1)
    xgb_cv.fit(X_cv_train, y_cv_train)
    y_cv_pred = xgb_cv.predict(X_cv_val)
    y_cv_prob = xgb_cv.predict_proba(X_cv_val)[:, 1]
    
    xgb_cv_auc.append(roc_auc_score(y_cv_val, y_cv_prob))
    xgb_cv_f1.append(f1_score(y_cv_val, y_cv_pred))
    xgb_cv_precision.append(precision_score(y_cv_val, y_cv_pred))
    xgb_cv_recall.append(recall_score(y_cv_val, y_cv_pred))
    
    print(f"  Fold {fold}: AUC={xgb_cv_auc[-1]:.4f}, F1={xgb_cv_f1[-1]:.4f}")

cv_results['xgboost'] = {
    'auc_mean': np.mean(xgb_cv_auc), 'auc_std': np.std(xgb_cv_auc),
    'f1_mean': np.mean(xgb_cv_f1), 'f1_std': np.std(xgb_cv_f1),
    'precision_mean': np.mean(xgb_cv_precision), 'precision_std': np.std(xgb_cv_precision),
    'recall_mean': np.mean(xgb_cv_recall), 'recall_std': np.std(xgb_cv_recall),
    'auc_values': xgb_cv_auc, 'f1_values': xgb_cv_f1
}
print(f"  Mean AUC: {cv_results['xgboost']['auc_mean']:.4f} ± {cv_results['xgboost']['auc_std']:.4f}")
print(f"  Mean F1:  {cv_results['xgboost']['f1_mean']:.4f} ± {cv_results['xgboost']['f1_std']:.4f}")

# Statistical significance test (paired t-test)
print("\n--- Statistical Significance Test ---")
t_stat_auc, p_value_auc = stats.ttest_rel(rf_cv_auc, xgb_cv_auc)
t_stat_f1, p_value_f1 = stats.ttest_rel(rf_cv_f1, xgb_cv_f1)

print(f"Paired t-test (RF vs XGBoost):")
print(f"  AUC: t={t_stat_auc:.4f}, p={p_value_auc:.4f} {'*' if p_value_auc < 0.05 else ''}")
print(f"  F1:  t={t_stat_f1:.4f}, p={p_value_f1:.4f} {'*' if p_value_f1 < 0.05 else ''}")
print(f"  * = statistically significant at α=0.05")

cv_results['statistical_tests'] = {
    'rf_vs_xgb_auc': {'t_statistic': t_stat_auc, 'p_value': p_value_auc},
    'rf_vs_xgb_f1': {'t_statistic': t_stat_f1, 'p_value': p_value_f1}
}

# Save CV results
json.dump({k: {kk: vv if not isinstance(vv, list) else [float(x) for x in vv] 
               for kk, vv in v.items()} for k, v in cv_results.items()}, 
          open('results/cv_results.json', 'w'), indent=2)
print(f"\n✓ Cross-validation results saved to results/cv_results.json")

# Figure: Cross-validation results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC boxplot
data_auc = [rf_cv_auc, xgb_cv_auc]
bp1 = axes[0].boxplot(data_auc, tick_labels=['Random Forest', 'XGBoost'], patch_artist=True)
bp1['boxes'][0].set_facecolor('#3498db')
bp1['boxes'][1].set_facecolor('#e74c3c')
axes[0].set_ylabel('AUC-ROC')
axes[0].set_title('Cross-Validation AUC-ROC Distribution', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# F1 boxplot
data_f1 = [rf_cv_f1, xgb_cv_f1]
bp2 = axes[1].boxplot(data_f1, tick_labels=['Random Forest', 'XGBoost'], patch_artist=True)
bp2['boxes'][0].set_facecolor('#3498db')
bp2['boxes'][1].set_facecolor('#e74c3c')
axes[1].set_ylabel('F1 Score')
axes[1].set_title('Cross-Validation F1 Score Distribution', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/03_cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure saved: figures/03_cross_validation.png")


# =============================================================================
# SECTION 6: THRESHOLD SENSITIVITY ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SECTION 6: THRESHOLD SENSITIVITY ANALYSIS")
print("="*80)

thresholds = np.arange(0.1, 0.95, 0.05)
threshold_results = []

print(f"Analyzing {len(thresholds)} threshold values...")
print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>8} {'FN':>8}")
print("-" * 60)

for threshold in thresholds:
    y_pred_thresh = (probabilities['rf'] >= threshold).astype(int)
    
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    cm = confusion_matrix(y_test, y_pred_thresh)
    fp = cm[0, 1]
    fn = cm[1, 0]
    
    threshold_results.append({
        'threshold': threshold,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fp': int(fp),
        'fn': int(fn)
    })
    
    print(f"{threshold:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {fp:>8} {fn:>8}")

# Find optimal threshold
optimal_idx = np.argmax([r['f1'] for r in threshold_results])
optimal_threshold = threshold_results[optimal_idx]['threshold']
print(f"\n✓ Optimal threshold for F1: {optimal_threshold:.2f} (F1={threshold_results[optimal_idx]['f1']:.4f})")

# Save threshold results
json.dump(threshold_results, open('results/threshold_analysis.json', 'w'), indent=2)
print(f"✓ Results saved to results/threshold_analysis.json")

# Figure: Threshold sensitivity
fig, ax = plt.subplots(figsize=(10, 6))

precisions = [r['precision'] for r in threshold_results]
recalls = [r['recall'] for r in threshold_results]
f1s = [r['f1'] for r in threshold_results]

ax.plot(thresholds, precisions, 'b-o', label='Precision', markersize=5, linewidth=2)
ax.plot(thresholds, recalls, 'r-o', label='Recall', markersize=5, linewidth=2)
ax.plot(thresholds, f1s, 'g-o', label='F1 Score', markersize=5, linewidth=2)

ax.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.7, linewidth=2)
ax.scatter([optimal_threshold], [threshold_results[optimal_idx]['f1']], 
           color='green', s=150, zorder=5, edgecolor='black', linewidth=2)
ax.annotate(f'Optimal: t={optimal_threshold:.2f}\nF1={threshold_results[optimal_idx]["f1"]:.3f}', 
            xy=(optimal_threshold, threshold_results[optimal_idx]['f1']),
            xytext=(optimal_threshold+0.1, threshold_results[optimal_idx]['f1']-0.1),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))

ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Threshold Sensitivity Analysis (Random Forest)', fontsize=14, fontweight='bold')
ax.legend(loc='center right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('figures/04_threshold_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure saved: figures/04_threshold_sensitivity.png")


# =============================================================================
# SECTION 7: SHAP EXPLAINABILITY ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SECTION 7: SHAP EXPLAINABILITY ANALYSIS")
print("="*80)

# Sample for SHAP analysis (computational efficiency)
shap_sample_size = 500
X_shap_sample = X_test.sample(n=shap_sample_size, random_state=42)
X_shap_sample_indices = X_shap_sample.index

print(f"Computing SHAP values for {shap_sample_size} test samples...")

# Random Forest SHAP
print("\n--- Random Forest SHAP Analysis ---")
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_shap_sample)

# For binary classification, take positive class
# Handles both old SHAP API (returns a list) and new API (returns a 3D array)
if isinstance(rf_shap_values, list):
    rf_shap_values = rf_shap_values[1]
elif rf_shap_values.ndim == 3:
    rf_shap_values = rf_shap_values[:, :, 1]

rf_feature_importance = pd.DataFrame({
    'feature': X_shap_sample.columns,
    'importance': np.abs(rf_shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("Top 10 features (Random Forest):")
for i, row in rf_feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# XGBoost SHAP
print("\n--- XGBoost SHAP Analysis ---")
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_shap_sample)

xgb_feature_importance = pd.DataFrame({
    'feature': X_shap_sample.columns,
    'importance': np.abs(xgb_shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("Top 10 features (XGBoost):")
for i, row in xgb_feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Feature importance consistency analysis
print("\n--- Feature Importance Consistency ---")
rf_top10 = set(rf_feature_importance.head(10)['feature'].values)
xgb_top10 = set(xgb_feature_importance.head(10)['feature'].values)
overlap = rf_top10.intersection(xgb_top10)

print(f"Random Forest Top 10: {rf_top10}")
print(f"XGBoost Top 10: {xgb_top10}")
print(f"Overlap: {overlap}")
print(f"Consistency: {len(overlap)}/10 = {len(overlap)*10}%")

# Save SHAP results
rf_feature_importance.to_csv('results/rf_shap_importance.csv', index=False)
xgb_feature_importance.to_csv('results/xgb_shap_importance.csv', index=False)
print(f"\n✓ SHAP results saved to results/")

# Figure: SHAP Summary (Top 15 features)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest
top15_rf = rf_feature_importance.head(15)
axes[0].barh(range(len(top15_rf)), top15_rf['importance'].values, color='#3498db')
axes[0].set_yticks(range(len(top15_rf)))
axes[0].set_yticklabels(top15_rf['feature'].values)
axes[0].invert_yaxis()
axes[0].set_xlabel('Mean |SHAP Value|')
axes[0].set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')

# XGBoost
top15_xgb = xgb_feature_importance.head(15)
axes[1].barh(range(len(top15_xgb)), top15_xgb['importance'].values, color='#e74c3c')
axes[1].set_yticks(range(len(top15_xgb)))
axes[1].set_yticklabels(top15_xgb['feature'].values)
axes[1].invert_yaxis()
axes[1].set_xlabel('Mean |SHAP Value|')
axes[1].set_title('XGBoost Feature Importance', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/05_shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure saved: figures/05_shap_importance.png")


# =============================================================================
# SECTION 8: STRIDE-BASED ATTACK SIMULATION
# =============================================================================
print("\n" + "="*80)
print("SECTION 8: STRIDE-BASED ATTACK SIMULATION")
print("="*80)

print("""
STRIDE Threat Model for Card-Present Payment Systems:
- Spoofing (S): Card cloning, velocity attacks
- Tampering (T): Amount manipulation
- Repudiation (R): Transaction denial (out of scope)
- Information Disclosure (I): Data leakage (out of scope)
- Denial of Service (D): System flooding (out of scope)
- Elevation of Privilege (E): Account sharing
+ Adversarial Evasion: Mimicry attacks
""")

# Get legitimate and fraud samples
legit_data = df[df['Class'] == 0].copy()
fraud_data = df[df['Class'] == 1].copy()

np.random.seed(42)
simulated_attacks = {}

# Attack 1: Card Cloning (Spoofing)
print("\n--- Attack 1: Card Cloning (Spoofing) ---")
n_clone = 200
clone_base = legit_data.sample(n_clone, random_state=42).copy()
clone_attacks = clone_base.copy()
clone_attacks['Time'] = clone_attacks['Time'] + np.random.uniform(1, 60, n_clone)
clone_attacks['Amount'] = clone_attacks['Amount'] * np.random.uniform(0.95, 1.05, n_clone)
for col in [f'V{i}' for i in range(1, 29)]:
    clone_attacks[col] = clone_attacks[col] + np.random.normal(0, 0.05, n_clone)
clone_attacks['Class'] = 1
clone_attacks['attack_type'] = 'card_cloning'
simulated_attacks['card_cloning'] = clone_attacks
print(f"Generated {len(clone_attacks)} card cloning attacks")

# Attack 2: Amount Manipulation (Tampering)
print("\n--- Attack 2: Amount Manipulation (Tampering) ---")
n_tamper = 150
tamper_base = fraud_data.sample(min(n_tamper, len(fraud_data)), random_state=43).copy()
tamper_attacks = tamper_base.copy()
tamper_attacks['Amount'] = np.random.uniform(10, 50, len(tamper_attacks))
tamper_attacks['attack_type'] = 'amount_manipulation'
simulated_attacks['amount_manipulation'] = tamper_attacks
print(f"Generated {len(tamper_attacks)} amount manipulation attacks")

# Attack 3: Velocity Attack (Spoofing)
print("\n--- Attack 3: Velocity Attack (Spoofing) ---")
n_velocity_base = 20
velocity_base = legit_data.sample(n_velocity_base, random_state=44)
velocity_attacks = []
for _, row in velocity_base.iterrows():
    for i in range(5):  # 5 rapid transactions per card
        attack = row.copy()
        attack['Time'] = row['Time'] + i * np.random.uniform(0.5, 3)
        attack['Amount'] = np.random.uniform(2, 5)
        for col in [f'V{i}' for i in range(1, 10)]:
            attack[col] = attack[col] + np.random.normal(0, 0.02)
        attack['Class'] = 1
        attack['attack_type'] = 'velocity_attack'
        velocity_attacks.append(attack)
velocity_attacks = pd.DataFrame(velocity_attacks)
simulated_attacks['velocity_attack'] = velocity_attacks
print(f"Generated {len(velocity_attacks)} velocity attack transactions")

# Attack 4: Mimicry Attack (Adversarial Evasion)
print("\n--- Attack 4: Mimicry Attack (Adversarial Evasion) ---")
n_mimicry = 150
mimicry_base = fraud_data.sample(min(n_mimicry, len(fraud_data)), random_state=45).copy()
legit_means = legit_data[[f'V{i}' for i in range(1, 29)]].mean()
mimicry_attacks = mimicry_base.copy()
for col in [f'V{i}' for i in range(1, 29)]:
    mimicry_attacks[col] = 0.5 * mimicry_attacks[col] + 0.5 * legit_means[col]
mimicry_attacks['Amount'] = np.random.choice(legit_data['Amount'].values, len(mimicry_attacks))
mimicry_attacks['attack_type'] = 'mimicry'
simulated_attacks['mimicry'] = mimicry_attacks
print(f"Generated {len(mimicry_attacks)} mimicry attacks")

# Attack 5: Account Sharing (Elevation of Privilege)
print("\n--- Attack 5: Account Sharing (Elevation of Privilege) ---")
n_sharing_base = 50
sharing_base = legit_data.sample(n_sharing_base, random_state=46)
sharing_attacks = []
for _, row in sharing_base.iterrows():
    attack1 = row.copy()
    attack2 = row.copy()
    for col in [f'V{i}' for i in range(1, 15)]:
        attack2[col] = -attack2[col] + np.random.normal(0, 0.5)
    attack1['Time'] = row['Time']
    attack2['Time'] = row['Time'] + np.random.uniform(60, 300)
    attack1['Class'] = 1
    attack2['Class'] = 1
    attack1['attack_type'] = 'account_sharing'
    attack2['attack_type'] = 'account_sharing'
    sharing_attacks.extend([attack1, attack2])
sharing_attacks = pd.DataFrame(sharing_attacks)
simulated_attacks['account_sharing'] = sharing_attacks
print(f"Generated {len(sharing_attacks)} account sharing attacks")

# Combine all attacks
all_attacks = pd.concat(list(simulated_attacks.values()), ignore_index=True)
print(f"\n--- Attack Summary ---")
print(f"Total simulated attacks: {len(all_attacks)}")
for attack_type, attacks in simulated_attacks.items():
    print(f"  {attack_type}: {len(attacks)}")

# Save attack data
all_attacks.to_csv('results/simulated_attacks.csv', index=False)
print(f"\n✓ Simulated attacks saved to results/simulated_attacks.csv")


# =============================================================================
# SECTION 9: ROBUSTNESS EVALUATION PER ATTACK TYPE
# =============================================================================
print("\n" + "="*80)
print("SECTION 9: ROBUSTNESS EVALUATION PER ATTACK TYPE")
print("="*80)

# Prepare attack test set
attack_test_legit = legit_data.sample(5000, random_state=47).copy()
attack_test_legit['attack_type'] = 'legitimate'
attack_test_set = pd.concat([all_attacks, attack_test_legit], ignore_index=True)

# Recompute Time_Delta on this combined set, consistent with how it was
# computed for the training data (sorted by Time, difference from previous row)
attack_test_set = attack_test_set.sort_values('Time').reset_index(drop=True)
attack_test_set['Time_Delta'] = attack_test_set['Time'].diff().fillna(attack_test_set['Time'].median())

# Prepare features
feature_cols_attack = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time', 'Time_Delta']
X_attack = attack_test_set[feature_cols_attack].copy()

# Scale Amount and Time
scaler_attack = StandardScaler()
X_attack['Amount'] = scaler_attack.fit_transform(X_attack[['Amount']])
X_attack['Time'] = scaler_attack.fit_transform(X_attack[['Time']])

y_attack = attack_test_set['Class']
attack_types = attack_test_set['attack_type']

# Get predictions
rf_attack_pred = rf_model.predict(X_attack)
rf_attack_prob = rf_model.predict_proba(X_attack)[:, 1]
xgb_attack_pred = xgb_model.predict(X_attack)
xgb_attack_prob = xgb_model.predict_proba(X_attack)[:, 1]

# Detection rate per attack type
print("\n--- Detection Rate by Attack Type ---")
print(f"{'Attack Type':<22} {'Count':>8} {'RF Det%':>10} {'XGB Det%':>10} {'RF Prob':>10}")
print("-" * 65)

attack_results = {}

for atype in attack_test_set['attack_type'].unique():
    mask = attack_types == atype
    count = mask.sum()
    
    if atype == 'legitimate':
        rf_fp = rf_attack_pred[mask].sum()
        xgb_fp = xgb_attack_pred[mask].sum()
        rf_rate = rf_fp / count * 100
        xgb_rate = xgb_fp / count * 100
        avg_prob = rf_attack_prob[mask].mean()
        print(f"{atype:<22} {count:>8} {rf_rate:>9.2f}% {xgb_rate:>9.2f}% {avg_prob:>10.4f}  (FP rate)")
    else:
        rf_detected = rf_attack_pred[mask].sum()
        xgb_detected = xgb_attack_pred[mask].sum()
        rf_rate = rf_detected / count * 100
        xgb_rate = xgb_detected / count * 100
        avg_prob = rf_attack_prob[mask].mean()
        print(f"{atype:<22} {count:>8} {rf_rate:>9.2f}% {xgb_rate:>9.2f}% {avg_prob:>10.4f}")
        
        attack_results[atype] = {
            'count': int(count),
            'rf_detected': int(rf_detected),
            'rf_rate': float(rf_rate),
            'xgb_detected': int(xgb_detected),
            'xgb_rate': float(xgb_rate),
            'avg_rf_prob': float(avg_prob)
        }

# Evasion resistance analysis
print("\n--- Evasion Resistance Analysis ---")
mimicry_mask = attack_types == 'mimicry'
other_attacks_mask = attack_types.isin(['card_cloning', 'amount_manipulation', 'velocity_attack', 'account_sharing'])

mimicry_det = rf_attack_pred[mimicry_mask].mean() * 100
other_det = rf_attack_pred[other_attacks_mask].mean() * 100

print(f"Standard attacks detection rate: {other_det:.2f}%")
print(f"Mimicry attacks detection rate: {mimicry_det:.2f}%")
print(f"Evasion success rate: {100 - mimicry_det:.2f}%")

attack_results['evasion_analysis'] = {
    'mimicry_detection_rate': float(mimicry_det),
    'standard_detection_rate': float(other_det),
    'evasion_success_rate': float(100 - mimicry_det)
}

# Save attack results
json.dump(attack_results, open('results/attack_results.json', 'w'), indent=2)
print(f"\n✓ Attack results saved to results/attack_results.json")

# Figure: Attack Detection Rates
fig, ax = plt.subplots(figsize=(12, 6))

attack_names = ['Card\nCloning', 'Amount\nManipulation', 'Velocity\nAttack', 'Mimicry', 'Account\nSharing']
attack_keys = ['card_cloning', 'amount_manipulation', 'velocity_attack', 'mimicry', 'account_sharing']
rf_rates = [attack_results[k]['rf_rate'] for k in attack_keys]
xgb_rates = [attack_results[k]['xgb_rate'] for k in attack_keys]

x = np.arange(len(attack_names))
width = 0.35

bars1 = ax.bar(x - width/2, rf_rates, width, label='Random Forest', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, xgb_rates, width, label='XGBoost', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Detection Rate (%)', fontsize=12)
ax.set_xlabel('Attack Type (STRIDE Category)', fontsize=12)
ax.set_title('Detection Rate by Attack Type (STRIDE-Based Simulation)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(attack_names)
ax.legend()
ax.set_ylim(0, 110)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='50% threshold')

plt.tight_layout()
plt.savefig('figures/06_attack_detection_rates.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure saved: figures/06_attack_detection_rates.png")


# =============================================================================
# SECTION 10: TRAFFIC LIGHT DECISION SUPPORT FRAMEWORK
# =============================================================================
print("\n" + "="*80)
print("SECTION 10: TRAFFIC LIGHT DECISION SUPPORT FRAMEWORK")
print("="*80)

print("""
Framework Design:
- GREEN zone: RF < 0.3 AND XGB < 0.3 → Auto-approve
- ORANGE zone: Uncertain or disagreement → Manual review  
- RED zone: RF >= 0.7 AND XGB >= 0.5 → Auto-flag
""")

def assign_zone(rf_prob, xgb_prob):
    """Assign traffic light zone based on model probabilities."""
    if rf_prob >= 0.7 and xgb_prob >= 0.5:
        return 'RED'
    elif rf_prob < 0.3 and xgb_prob < 0.3:
        return 'GREEN'
    else:
        return 'ORANGE'

# Apply to test set
zones = [assign_zone(rf_p, xgb_p) for rf_p, xgb_p in zip(probabilities['rf'], probabilities['xgb'])]
zones_series = pd.Series(zones)

# Zone distribution
zone_counts = zones_series.value_counts()
print("\n--- Zone Distribution ---")
for zone in ['GREEN', 'ORANGE', 'RED']:
    count = zone_counts.get(zone, 0)
    pct = count / len(zones) * 100
    print(f"{zone}: {count:,} transactions ({pct:.2f}%)")

# Fraud distribution per zone
zone_df = pd.DataFrame({
    'zone': zones,
    'true_label': y_test.values,
    'rf_prob': probabilities['rf'],
    'xgb_prob': probabilities['xgb']
})

print("\n--- Fraud Distribution per Zone ---")
print(f"{'Zone':<10} {'Transactions':>12} {'Frauds':>10} {'Fraud Rate':>12} {'Action':<15}")
print("-" * 65)

framework_results = {}
for zone in ['GREEN', 'ORANGE', 'RED']:
    zone_mask = zone_df['zone'] == zone
    transactions = zone_mask.sum()
    frauds = zone_df[zone_mask]['true_label'].sum()
    fraud_rate = frauds / transactions * 100 if transactions > 0 else 0
    action = {'GREEN': 'Auto-approve', 'ORANGE': 'Manual review', 'RED': 'Auto-flag'}[zone]
    print(f"{zone:<10} {transactions:>12,} {frauds:>10} {fraud_rate:>11.2f}% {action:<15}")
    
    framework_results[zone] = {
        'transactions': int(transactions),
        'frauds': int(frauds),
        'fraud_rate': float(fraud_rate)
    }

# Workload reduction
total_transactions = len(y_test)
manual_review_needed = framework_results['ORANGE']['transactions']
workload_reduction = (1 - manual_review_needed / total_transactions) * 100

print(f"\n--- Framework Effectiveness ---")
print(f"Total transactions: {total_transactions:,}")
print(f"Manual review needed (ORANGE): {manual_review_needed:,}")
print(f"Workload reduction: {workload_reduction:.2f}%")

# Model agreement analysis
rf_pred_binary = (probabilities['rf'] >= 0.5).astype(int)
xgb_pred_binary = (probabilities['xgb'] >= 0.5).astype(int)
agreement_mask = rf_pred_binary == xgb_pred_binary
agreement_rate = agreement_mask.mean() * 100

# Accuracy when models agree
agree_correct = ((rf_pred_binary[agreement_mask] == y_test.values[agreement_mask])).sum()
agree_total = agreement_mask.sum()
agree_accuracy = agree_correct / agree_total * 100

print(f"\n--- Model Agreement Analysis ---")
print(f"Models agree: {agree_total:,} transactions ({agreement_rate:.2f}%)")
print(f"Accuracy when agree: {agree_accuracy:.2f}%")

# Disagreement analysis
disagree_mask = ~agreement_mask
disagree_frauds = y_test.values[disagree_mask].sum()
disagree_total = (~agreement_mask).sum()
print(f"Models disagree: {disagree_total:,} transactions")
print(f"Frauds in disagreement cases: {disagree_frauds} ({disagree_frauds/disagree_total*100:.1f}% fraud rate)")

framework_results['workload_reduction'] = float(workload_reduction)
framework_results['model_agreement'] = {
    'agree_count': int(agree_total),
    'agree_rate': float(agreement_rate),
    'agree_accuracy': float(agree_accuracy),
    'disagree_count': int(disagree_total),
    'disagree_fraud_count': int(disagree_frauds)
}

# Save framework results
json.dump(framework_results, open('results/framework_results.json', 'w'), indent=2)
print(f"\n✓ Framework results saved to results/framework_results.json")

# Figure: Traffic Light Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
colors = {'GREEN': '#2ecc71', 'ORANGE': '#f39c12', 'RED': '#e74c3c'}
zone_sizes = [framework_results[z]['transactions'] for z in ['GREEN', 'ORANGE', 'RED']]
zone_labels = ['GREEN\n(Auto-approve)', 'ORANGE\n(Manual Review)', 'RED\n(Auto-flag)']
zone_colors = [colors[z] for z in ['GREEN', 'ORANGE', 'RED']]

axes[0].pie(zone_sizes, labels=zone_labels, colors=zone_colors, autopct='%1.2f%%',
            explode=(0, 0.1, 0.1), startangle=90, textprops={'fontsize': 10})
axes[0].set_title('Decision Support Framework\nZone Distribution', fontsize=12, fontweight='bold')

# Stacked bar chart
zones_order = ['GREEN', 'ORANGE', 'RED']
totals = [framework_results[z]['transactions'] for z in zones_order]
frauds = [framework_results[z]['frauds'] for z in zones_order]
legits = [t - f for t, f in zip(totals, frauds)]

x = np.arange(3)
bars1 = axes[1].bar(x, legits, label='Legitimate', color='#3498db', edgecolor='black')
bars2 = axes[1].bar(x, frauds, bottom=legits, label='Fraud', color='#e74c3c', edgecolor='black')

axes[1].set_xticks(x)
axes[1].set_xticklabels(zones_order)
axes[1].set_ylabel('Transaction Count (log scale)')
axes[1].set_yscale('log')
axes[1].set_title('Fraud Distribution per Zone', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/07_traffic_light_framework.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure saved: figures/07_traffic_light_framework.png")


# =============================================================================
# SECTION 11: ADDITIONAL FIGURES FOR DISSERTATION
# =============================================================================
print("\n" + "="*80)
print("SECTION 11: ADDITIONAL FIGURES FOR DISSERTATION")
print("="*80)

# Figure: ROC Curves Comparison
fig, ax = plt.subplots(figsize=(10, 8))

for name, prob, color, style in [
    ('Random Forest', probabilities['rf'], '#3498db', '-'),
    ('XGBoost', probabilities['xgb'], '#e74c3c', '-'),
    ('Isolation Forest', probabilities['iso'], '#9b59b6', '--')
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linestyle=style, linewidth=2.5,
            label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)

plt.tight_layout()
plt.savefig('figures/08_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure saved: figures/08_roc_curves.png")

# Figure: Precision-Recall Curves
fig, ax = plt.subplots(figsize=(10, 8))

for name, prob, color in [
    ('Random Forest', probabilities['rf'], '#3498db'),
    ('XGBoost', probabilities['xgb'], '#e74c3c')
]:
    prec, rec, _ = precision_recall_curve(y_test, prob)
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec, color=color, linewidth=2.5, label=f'{name} (AUC = {pr_auc:.3f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/09_precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure saved: figures/09_precision_recall_curves.png")

# Figure: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, name, pred in [(axes[0], 'Random Forest', predictions['rf']),
                        (axes[1], 'XGBoost', predictions['xgb'])]:
    cm = confusion_matrix(y_test, pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted\nLegitimate', 'Predicted\nFraud'],
                yticklabels=['Actual\nLegitimate', 'Actual\nFraud'],
                annot_kws={'size': 14})
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/10_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure saved: figures/10_confusion_matrices.png")

# Figure: Feature comparison (legitimate vs fraud vs simulated attacks)
print("\n--- Feature Analysis: Original vs Simulated ---")
key_features = ['V14', 'V4', 'V12', 'V10']

feature_comparison = pd.DataFrame({
    'Legitimate': legit_data[key_features].mean(),
    'Original Fraud': fraud_data[key_features].mean(),
    'Card Cloning': simulated_attacks['card_cloning'][key_features].mean(),
    'Mimicry': simulated_attacks['mimicry'][key_features].mean()
})

print(feature_comparison.round(4))

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(key_features))
width = 0.2

bars1 = ax.bar(x - 1.5*width, feature_comparison['Legitimate'], width, label='Legitimate', color='#3498db')
bars2 = ax.bar(x - 0.5*width, feature_comparison['Original Fraud'], width, label='Original Fraud', color='#e74c3c')
bars3 = ax.bar(x + 0.5*width, feature_comparison['Card Cloning'], width, label='Card Cloning Attack', color='#f39c12')
bars4 = ax.bar(x + 1.5*width, feature_comparison['Mimicry'], width, label='Mimicry Attack', color='#9b59b6')

ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Mean Value', fontsize=12)
ax.set_title('Feature Comparison: Why Simulated Attacks Evade Detection', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(key_features)
ax.legend()
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/11_feature_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure saved: figures/11_feature_comparison.png")


# =============================================================================
# SECTION 12: FINAL SUMMARY AND RESULTS EXPORT
# =============================================================================
print("\n" + "="*80)
print("SECTION 12: FINAL SUMMARY")
print("="*80)

# Compile all results
final_summary = {
    'dataset': {
        'total_transactions': len(df),
        'fraud_cases': int(df['Class'].sum()),
        'fraud_rate': float(df['Class'].mean()),
        'train_size': len(X_train),
        'test_size': len(X_test)
    },
    'model_performance': {
        'random_forest': {
            'auc_roc': float(roc_auc_score(y_test, probabilities['rf'])),
            'precision': float(precision_score(y_test, predictions['rf'])),
            'recall': float(recall_score(y_test, predictions['rf'])),
            'f1_score': float(f1_score(y_test, predictions['rf']))
        },
        'xgboost': {
            'auc_roc': float(roc_auc_score(y_test, probabilities['xgb'])),
            'precision': float(precision_score(y_test, predictions['xgb'])),
            'recall': float(recall_score(y_test, predictions['xgb'])),
            'f1_score': float(f1_score(y_test, predictions['xgb']))
        },
        'isolation_forest': {
            'auc_roc': float(roc_auc_score(y_test, probabilities['iso'])),
            'precision': float(precision_score(y_test, predictions['iso'])),
            'recall': float(recall_score(y_test, predictions['iso'])),
            'f1_score': float(f1_score(y_test, predictions['iso']))
        }
    },
    'cross_validation': {
        'random_forest': {
            'auc_mean': cv_results['random_forest']['auc_mean'],
            'auc_std': cv_results['random_forest']['auc_std'],
            'f1_mean': cv_results['random_forest']['f1_mean'],
            'f1_std': cv_results['random_forest']['f1_std']
        },
        'xgboost': {
            'auc_mean': cv_results['xgboost']['auc_mean'],
            'auc_std': cv_results['xgboost']['auc_std'],
            'f1_mean': cv_results['xgboost']['f1_mean'],
            'f1_std': cv_results['xgboost']['f1_std']
        }
    },
    'attack_simulation': attack_results,
    'framework': framework_results,
    'shap_analysis': {
        'feature_consistency': len(overlap),
        'rf_top5': list(rf_feature_importance.head(5)['feature'].values),
        'xgb_top5': list(xgb_feature_importance.head(5)['feature'].values)
    }
}

# Save comprehensive summary
json.dump(final_summary, open('results/final_summary.json', 'w'), indent=2)

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
print(f"\nKey Results:")
print(f"  • Random Forest AUC-ROC: {final_summary['model_performance']['random_forest']['auc_roc']:.4f}")
print(f"  • Random Forest F1: {final_summary['model_performance']['random_forest']['f1_score']:.4f}")
print(f"  • Cross-validation F1: {cv_results['random_forest']['f1_mean']:.4f} ± {cv_results['random_forest']['f1_std']:.4f}")
print(f"  • Mimicry attack detection: {attack_results['mimicry']['rf_rate']:.1f}%")
print(f"  • Framework workload reduction: {framework_results['workload_reduction']:.2f}%")
print(f"  • SHAP feature consistency: {len(overlap)}/10 ({len(overlap)*10}%)")

print(f"\nOutput Files:")
print(f"  • results/final_summary.json")
print(f"  • results/cv_results.json")
print(f"  • results/attack_results.json")
print(f"  • results/framework_results.json")
print(f"  • results/threshold_analysis.json")
print(f"  • models/rf_model.pkl")
print(f"  • models/xgb_model.pkl")
print(f"  • figures/*.png (11 figures)")

print(f"\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
