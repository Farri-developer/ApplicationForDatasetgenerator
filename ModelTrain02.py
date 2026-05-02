# ============================================================
#  STRESS LEVEL CLASSIFICATION - COMPLETE TRAINING SCRIPT
#  Features: EEG (4 channels x 6 bands) + PPG/HRV + BP
#  Target: Stress_Label (Low / Medium / High)
# ============================================================

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from datetime import datetime
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
#  CONFIG  — sirf yahan path badlein
# ============================================================
DATA_FILE     = r"D:\Path\merged_dataset.csv"   # apna path yahan daalein
OUTPUT_FOLDER = "Stress_Results"

# leakage wale columns jo drop karne hain
DROP_COLS = ["Stress_Score"]            # sirf ye drop hoga

# ============================================================
#  HELPER
# ============================================================
def banner(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def ts():
    return datetime.now().strftime("%H:%M:%S")

# ============================================================
#  STEP 1 — DATA LOAD
# ============================================================
banner("STEP 1 — DATA LOAD")

if not os.path.exists(DATA_FILE):
    print(f"❌  File not found: {DATA_FILE}")
    sys.exit(1)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print(f"[{ts()}] ✅ Loaded  → {df.shape[0]} rows × {df.shape[1]} cols")
print(f"[{ts()}]    Columns : {df.columns.tolist()}")

if "Stress_Label" not in df.columns:
    print("❌  'Stress_Label' column missing!")
    sys.exit(1)

print(f"\n[{ts()}]  Class distribution:")
print(df["Stress_Label"].value_counts().to_string())

# ============================================================
#  STEP 2 — CLEAN
# ============================================================
banner("STEP 2 — CLEANING")

df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# numeric conversion + NaN fill
X_raw = df.drop("Stress_Label", axis=1)
for col in X_raw.columns:
    X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
X_raw.fillna(X_raw.mean(), inplace=True)

y_raw = df["Stress_Label"].astype(str).str.strip()

print(f"[{ts()}] ✅ Features after clean: {X_raw.shape[1]}")
print(f"[{ts()}]    NaN remaining: {X_raw.isnull().sum().sum()}")

# encode labels → 0,1,2
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
print(f"[{ts()}]    Classes : {class_names}")

# ============================================================
#  STEP 3 — TRAIN / TEST SPLIT
# ============================================================
banner("STEP 3 — TRAIN/TEST SPLIT  (80/20, stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"[{ts()}] ✅ Train : {len(X_train)} samples")
print(f"[{ts()}]    Test  : {len(X_test)}  samples")
print(f"[{ts()}]    Features : {X_train.shape[1]}")

# ============================================================
#  STEP 4 — TRAIN (Random Forest)
# ============================================================
banner("STEP 4 — MODEL TRAINING")

rf = RandomForestClassifier(
    n_estimators   = 300,
    max_depth       = None,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    class_weight    = 'balanced',
    random_state    = 42,
    n_jobs          = -1
)

print(f"[{ts()}] 🚀 Training Random Forest (300 trees)...")
rf.fit(X_train, y_train)
print(f"[{ts()}] ✅ Training complete!")

# ============================================================
#  STEP 5 — EVALUATION (Full)
# ============================================================
banner("STEP 5 — COMPLETE EVALUATION")

y_pred       = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

acc  = accuracy_score (y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score   (y_test, y_pred, average='weighted')
f1   = f1_score       (y_test, y_pred, average='weighted')

print(f"\n{'':4}{'Metric':<15} {'Score':>10}")
print(f"{'':4}{'-'*26}")
print(f"{'':4}{'Accuracy':<15} {acc*100:>9.2f}%")
print(f"{'':4}{'Precision':<15} {prec*100:>9.2f}%")
print(f"{'':4}{'Recall':<15} {rec*100:>9.2f}%")
print(f"{'':4}{'F1-Score':<15} {f1*100:>9.2f}%")

print(f"\n[{ts()}]  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"\n[{ts()}]  Per-Class Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ---- Cross-Validation (5-fold) ----
print(f"[{ts()}] 🔄 Running 5-Fold Cross-Validation on training set...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
print(f"[{ts()}]    CV Folds   : {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"[{ts()}]    CV Mean    : {cv_scores.mean()*100:.2f}%")
print(f"[{ts()}]    CV Std Dev : {cv_scores.std()*100:.2f}%")

# ============================================================
#  STEP 6 — FEATURE IMPORTANCE
# ============================================================
banner("STEP 6 — FEATURE IMPORTANCE")

feat_df = pd.DataFrame({
    'Feature'   : X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(f"\n  Top-15 Features:")
print(feat_df.head(15).to_string(index=False))

# ============================================================
#  STEP 7 — VISUALIZATIONS
# ============================================================
banner("STEP 7 — SAVING VISUALIZATIONS")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 7a  Confusion Matrix ---
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/01_confusion_matrix.png", dpi=200)
plt.close()
print(f"[{ts()}] ✅ 01_confusion_matrix.png")

# --- 7b  Metrics Bar ---
fig, ax = plt.subplots(figsize=(7, 5))
metrics  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values   = [acc, prec, rec, f1]
colors   = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFA07A']
bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
ax.set_ylim(0, 1.1)
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f"{b.get_height()*100:.2f}%", ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/02_metrics.png", dpi=200)
plt.close()
print(f"[{ts()}] ✅ 02_metrics.png")

# --- 7c  Feature Importance (top 15) ---
top15 = feat_df.head(15).sort_values('Importance')
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top15['Feature'], top15['Importance'], color='#3498db', edgecolor='black')
ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/03_feature_importance.png", dpi=200)
plt.close()
print(f"[{ts()}] ✅ 03_feature_importance.png")

# --- 7d  ROC Curves ---
y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ['#e74c3c', '#2ecc71', '#3498db']
for i, (cls, col) in enumerate(zip(class_names, colors_roc)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=col, label=f'{cls} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1],'--', color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves per Class', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/04_roc_curves.png", dpi=200)
plt.close()
print(f"[{ts()}] ✅ 04_roc_curves.png")

# --- 7e  Cross-Val Scores ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 6), cv_scores * 100, color='#9b59b6', edgecolor='black')
ax.axhline(cv_scores.mean()*100, color='red', linestyle='--',
           label=f'Mean = {cv_scores.mean()*100:.2f}%')
ax.set_title('5-Fold Cross-Validation Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy (%)')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/05_cross_validation.png", dpi=200)
plt.close()
print(f"[{ts()}] ✅ 05_cross_validation.png")

# ============================================================
#  STEP 8 — SAVE MODEL + REPORT
# ============================================================
banner("STEP 8 — SAVE MODEL & REPORT")

joblib.dump(rf, f"{OUTPUT_FOLDER}/stress_rf_model.pkl")
joblib.dump(le, f"{OUTPUT_FOLDER}/label_encoder.pkl")
joblib.dump(X_train.columns.tolist(), f"{OUTPUT_FOLDER}/feature_names.pkl")

feat_df.to_csv(f"{OUTPUT_FOLDER}/feature_importance.csv", index=False)

# text report
report_path = f"{OUTPUT_FOLDER}/FINAL_REPORT.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("STRESS CLASSIFICATION — RANDOM FOREST REPORT\n")
    f.write(f"Generated : {datetime.now()}\n")
    f.write("="*55 + "\n\n")
    f.write(f"Dataset    : {DATA_FILE}\n")
    f.write(f"Total rows : {len(df)}\n")
    f.write(f"Features   : {X_train.shape[1]}\n")
    f.write(f"Classes    : {list(class_names)}\n")
    f.write(f"Train / Test split : 80 / 20\n\n")
    f.write("--- METRICS ---\n")
    f.write(f"Accuracy  : {acc*100:.2f}%\n")
    f.write(f"Precision : {prec*100:.2f}%\n")
    f.write(f"Recall    : {rec*100:.2f}%\n")
    f.write(f"F1-Score  : {f1*100:.2f}%\n\n")
    f.write("--- CROSS-VALIDATION (5-Fold) ---\n")
    f.write(f"Mean  : {cv_scores.mean()*100:.2f}%\n")
    f.write(f"Std   : {cv_scores.std()*100:.2f}%\n\n")
    f.write("--- CONFUSION MATRIX ---\n")
    f.write(str(cm) + "\n\n")
    f.write("--- CLASSIFICATION REPORT ---\n")
    f.write(classification_report(y_test, y_pred, target_names=class_names))

print(f"[{ts()}] ✅ Model  → {OUTPUT_FOLDER}/stress_rf_model.pkl")
print(f"[{ts()}] ✅ Report → {report_path}")

# ============================================================
#  DONE
# ============================================================
banner("✅  ALL DONE")
print(f"\n  Folder : {OUTPUT_FOLDER}/")
print(f"  Files  :")
for f in sorted(os.listdir(OUTPUT_FOLDER)):
    print(f"    📄 {f}")