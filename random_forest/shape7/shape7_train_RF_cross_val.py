import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime
import random
import joblib

# ---------------- CONFIG ------------------
INPUT_CSV = "shape7_dataset_labeled.csv"
OUTPUT_DIR = "rf_outputs"
CORR_THRESHOLD = 0.9
RANDOM_STATE = 56

# Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TXT_LOG = os.path.join(OUTPUT_DIR, f"rf_results_{TIMESTAMP}.txt")
MODEL_PATH = os.path.join(OUTPUT_DIR, f"rf_model_{TIMESTAMP}.joblib")

# --------------- LOAD DATA ----------------
df = pd.read_csv(INPUT_CSV)
X = df.drop(columns=["child_id", "label"])
y = df["label"]

# --------------- REMOVE CORRELATED FEATURES ----------------
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESHOLD)]
X.drop(columns=to_drop, inplace=True)

# --------------- SPLIT DATA ----------------
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=RANDOM_STATE)

# --------------- AUGMENT TRAIN SET ----------------

class_counts = y_train.value_counts()
TARGET_PER_CLASS = class_counts.max()
def augment_class_rows(X_cls, num_needed):
    augmented = []
    for _ in range(num_needed):
        row = X_cls.sample(n=1, replace=True, random_state=random.randint(0, 10000)).values.flatten()
        noise = np.random.normal(0, 0.05, size=row.shape)  # 5% noise
        new_row = row * (1 + noise)
        augmented.append(new_row)
    return np.array(augmented)

X_train_aug, y_train_aug = X_train.copy(), y_train.copy()
for cls in sorted(y_train.unique()):
    X_cls = X_train[y_train == cls]
    count = len(X_cls)
    if count < TARGET_PER_CLASS:
        needed = TARGET_PER_CLASS - count
        new_rows = augment_class_rows(X_cls, needed)
        X_train_aug = pd.concat([X_train_aug, pd.DataFrame(new_rows, columns=X_train.columns)], ignore_index=True)
        y_train_aug = pd.concat([y_train_aug, pd.Series([cls]*needed)], ignore_index=True)

# --------------- GRID SEARCH ----------------
param_grid = {
    'classifier__n_estimators': [325],
    'classifier__max_depth': [20, 30],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
])

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train_aug, y_train_aug)

# Save model
joblib.dump(grid.best_estimator_, MODEL_PATH)

# --------------- EVALUATION ----------------
y_pred = grid.predict(X_test)
acc = accuracy_score(y_test, y_pred)
soft_acc = np.mean(np.abs(y_test - y_pred) <= 1)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# --------------- OUTPUT TO TXT ----------------
with open(TXT_LOG, 'w') as f:
    f.write("=== Random Forest Results ===\n")
    f.write(f"Timestamp: {TIMESTAMP}\n\n")
    f.write(f"Dropped correlated features: {to_drop}\n\n")
    f.write(f"Best Params: {grid.best_params_}\n\n")
    f.write(f"Model saved to: {MODEL_PATH}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write(f"\nAccuracy: {acc:.4f}\n")
    f.write(f"Soft Accuracy (|pred - true| <= 1): {soft_acc:.4f}\n\n")

    # Feature importances
    best_rf = grid.best_estimator_.named_steps['classifier']
    importances = best_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    f.write("\nTop 10 Feature Importances:\n")
    for i in sorted_idx[:10]:
        f.write(f"{X_train.columns[i]}: {importances[i]:.4f}\n")

# --------------- PLOTS ----------------
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"conf_matrix_{TIMESTAMP}.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x=[X_train.columns[i] for i in sorted_idx[:10]], y=importances[sorted_idx[:10]])
plt.title("Top 10 Feature Importances")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"feature_importance_{TIMESTAMP}.png"))
plt.close()

# Save classification report as heatmap
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :3]  # remove avg rows
plt.figure(figsize=(10, 6))
sns.heatmap(report_df, annot=True, cmap="YlGnBu")
plt.title("Classification Report (Precision/Recall/F1)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"class_report_{TIMESTAMP}.png"))
plt.close()
