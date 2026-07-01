import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: CREATE THE DATASET
# Based on EU AI Act risk classification logic
# ─────────────────────────────────────────────
np.random.seed(42)
n = 300

data = {
    'affects_human_rights':     np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'used_in_critical_sector':  np.random.choice([0, 1], n, p=[0.5, 0.5]),
    'makes_autonomous_decisions': np.random.choice([0, 1], n, p=[0.45, 0.55]),
    'processes_biometric_data': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'has_human_oversight':      np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'used_on_vulnerable_groups': np.random.choice([0, 1], n, p=[0.65, 0.35]),
    'transparency_available':   np.random.choice([0, 1], n, p=[0.35, 0.65]),
    'explainability_score':     np.random.randint(1, 11, n),
    'data_quality_score':       np.random.randint(1, 11, n),
    'deployed_in_production':   np.random.choice([0, 1], n, p=[0.45, 0.55]),
}

df = pd.DataFrame(data)

# Risk label logic based on EU AI Act principles
def assign_risk(row):
    score = 0
    if row['affects_human_rights']:       score += 3
    if row['used_in_critical_sector']:    score += 2
    if row['makes_autonomous_decisions']: score += 2
    if row['processes_biometric_data']:   score += 2
    if not row['has_human_oversight']:    score += 2
    if row['used_on_vulnerable_groups']:  score += 2
    if not row['transparency_available']: score += 1
    if row['explainability_score'] < 5:   score += 1
    if row['data_quality_score'] < 5:     score += 1
    if row['deployed_in_production']:     score += 1

    if score >= 10:   return 'High Risk'
    elif score >= 6:  return 'Medium Risk'
    else:             return 'Low Risk'

df['risk_level'] = df.apply(assign_risk, axis=1)

print("=" * 55)
print("  AI RISK CLASSIFIER — EU AI ACT BASED")
print("  ISTQB CT-AI Hands-on Exercise")
print("=" * 55)

print("\n📊 DATASET OVERVIEW")
print(f"   Total AI systems in dataset : {len(df)}")
print(f"   Features used               : {len(df.columns) - 1}")
print("\n   Risk level distribution:")
for label, count in df['risk_level'].value_counts().items():
    pct = count / len(df) * 100
    bar = '█' * int(pct / 3)
    print(f"   {label:<14} {bar:<18} {count} ({pct:.1f}%)")

# ─────────────────────────────────────────────
# STEP 2: PREPARE DATA
# ─────────────────────────────────────────────
le = LabelEncoder()
df['risk_encoded'] = le.fit_transform(df['risk_level'])

X = df.drop(['risk_level', 'risk_encoded'], axis=1)
y = df['risk_encoded']

# Split 1: Separate test set (20%) — locked away, never touched during training
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Split 2: From remaining 80%, split into train (75%) and validation (25%)
# Final ratio: 60% train, 20% validation, 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"\n📂 DATA SPLIT")
print(f"   Training set   : {len(X_train)} systems (60%) — used to TEACH the model")
print(f"   Validation set : {len(X_val)} systems (20%) — used to TUNE the model")
print(f"   Test set       : {len(X_test)} systems (20%) — used to EVALUATE the model")

# ─────────────────────────────────────────────
# STEP 3: TRAIN — try different n_estimators
# (this is tuning — like exploratory testing)
# ─────────────────────────────────────────────
print("\n🔧 TUNING THE MODEL (on validation set)")
print("   Testing different settings — like a tester trying different configs:")
print()

results = []
for n_trees in [10, 25, 50, 100, 200]:
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    results.append((n_trees, val_acc, model))
    bar = '█' * int(val_acc * 30)
    print(f"   {n_trees:>4} trees  |  {bar:<30}  {val_acc:.1%}")

# Pick best model
best = max(results, key=lambda x: x[1])
best_n, best_val_acc, best_model = best
print(f"\n   ✅ Best setting: {best_n} trees → Validation accuracy: {best_val_acc:.1%}")

# ─────────────────────────────────────────────
# STEP 4: FINAL TEST (on locked test set)
# (this is like regression testing — final check)
# ─────────────────────────────────────────────
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
labels = le.classes_

print("\n" + "=" * 55)
print("🎯 FINAL TEST RESULTS (on unseen test data)")
print("=" * 55)
print(f"\n   Validation accuracy : {best_val_acc:.1%}  (seen during tuning)")
print(f"   Test accuracy       : {test_acc:.1%}  (never seen before)")

gap = abs(test_acc - best_val_acc)
if gap < 0.05:
    verdict = "✅ GOOD — model generalises well (gap < 5%)"
else:
    verdict = "⚠️  WARNING — possible overfitting (gap > 5%)"
print(f"   Gap                 : {gap:.1%}  → {verdict}")

print("\n📋 CLASSIFICATION REPORT")
print("   (How well did the model classify each risk level?)\n")
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
for label in labels:
    r = report[label]
    print(f"   {label:<14}  Precision: {r['precision']:.0%}  Recall: {r['recall']:.0%}  F1: {r['f1-score']:.0%}  (n={int(r['support'])})")

# ─────────────────────────────────────────────
# STEP 5: GENERATE CHARTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#F8F9FA')
colors_brand = ['#EF4444', '#F59E0B', '#10B981']

# Chart 1: Tuning curve
ax1 = axes[0]
ax1.set_facecolor('white')
ns = [r[0] for r in results]
accs = [r[1] for r in results]
ax1.plot(ns, accs, 'o-', color='#0F1B35', linewidth=2.5, markersize=8, markerfacecolor='#C9935A')
ax1.axhline(y=test_acc, color='#EF4444', linestyle='--', linewidth=1.5, label=f'Test accuracy: {test_acc:.1%}')
ax1.set_title('Tuning: Validation Accuracy\nvs Number of Trees', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Number of Trees', fontsize=10)
ax1.set_ylabel('Accuracy', fontsize=10)
ax1.set_ylim(0.5, 1.0)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Chart 2: Confusion matrix
ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='white', cbar=False)
ax2.set_title('Confusion Matrix\n(Predicted vs Actual)', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Predicted Risk Level', fontsize=10)
ax2.set_ylabel('Actual Risk Level', fontsize=10)
ax2.tick_params(axis='x', rotation=15)
ax2.tick_params(axis='y', rotation=0)

# Chart 3: Feature importance
ax3 = axes[2]
ax3.set_facecolor('white')
importances = best_model.feature_importances_
feat_names = [c.replace('_', ' ').title() for c in X.columns]
sorted_idx = np.argsort(importances)
colors_imp = ['#C9935A' if importances[i] == max(importances) else '#0F1B35' for i in sorted_idx]
bars = ax3.barh(range(len(sorted_idx)), importances[sorted_idx], color=colors_imp, alpha=0.85, height=0.6)
ax3.set_yticks(range(len(sorted_idx)))
ax3.set_yticklabels([feat_names[i] for i in sorted_idx], fontsize=8)
ax3.set_title('Feature Importance\n(What drives the risk level?)', fontsize=12, fontweight='bold', pad=12)
ax3.set_xlabel('Importance Score', fontsize=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(True, alpha=0.2, axis='x')

plt.suptitle('AI Risk Classifier — EU AI Act Based | ISTQB CT-AI Exercise',
             fontsize=13, fontweight='bold', y=1.02, color='#0F1B35')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/ai_risk_classifier_results.png',
            dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()

print("\n📊 Charts saved successfully")
print("\n" + "=" * 55)
print("✅ EXERCISE COMPLETE")
print("=" * 55)
