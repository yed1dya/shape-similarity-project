import winsound

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime
import joblib

# ---------------- CONFIG ------------------
SHAPE = 9
INPUT_CSV = "shape9_features_20250805_152433.csv"
OUTPUT_DIR = f"rf_shape{SHAPE}_outputs"
CORR_THRESHOLD = 0.9
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

start_time = datetime.now()
TIMESTAMP = start_time.strftime("%Y%m%d_%H%M%S")
print(f"start time: {TIMESTAMP}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TXT_LOG = os.path.join(OUTPUT_DIR, f"rf_{SHAPE}_results_{TIMESTAMP}.txt")
MODEL_PATH = os.path.join(OUTPUT_DIR, f"rf_{SHAPE}_model_{TIMESTAMP}.joblib")

# --------------- LOAD DATA ----------------
df = pd.read_csv(INPUT_CSV)
X = df.drop(columns=["child_id", "label"])
y = df["label"]

# --------------- REMOVE CORRELATED FEATURES ----------------
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESHOLD)]
X.drop(columns=to_drop, inplace=True)
print(f"class distribution: {y.value_counts()}")

# --------------- SPLIT DATA ----------------
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=RANDOM_STATE)

# --------------- SET TARGET PER CLASS ----------------
class_counts = y_train.value_counts()
TARGET_PER_CLASS = class_counts.max()
print(f"TARGET_PER_CLASS={TARGET_PER_CLASS}")

# --- Save and plot class distribution ---
class_dist_path = os.path.join(OUTPUT_DIR, f"class_distribution_{TIMESTAMP}.png")

plt.figure(figsize=(10, 5))
sns.barplot(x=class_counts.index.astype(str), y=class_counts.values)
plt.title("Training Set Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(class_dist_path)
plt.close()


def soft_accuracy(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) <= 1)


# --------------- AUGMENT TRAIN SET ----------------
def augment_class_rows(X_cls, num_needed):
    augmented = []
    for i in range(num_needed):
        row = X_cls.sample(n=1, replace=True, random_state=RANDOM_STATE + i).values.flatten()
        noise = np.random.normal(0, 0.05, size=row.shape)
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
    'classifier__n_estimators': [275],
    'classifier__max_depth': [25],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1]
}
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
])
soft_scorer = make_scorer(soft_accuracy, greater_is_better=True)
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

# Save val and test class counts
val_class_counts = y_val.value_counts()
test_class_counts = y_test.value_counts()
with open(TXT_LOG, 'w') as f:
    f.write(f"Shape {SHAPE} Random Forest Classifier\n")
    f.write("Training set class distribution:\n")
    f.write(class_counts.to_string())
    f.write("\nValidation set class distribution:\n")
    f.write(val_class_counts.to_string())
    f.write("\nTest set class distribution:\n")
    f.write(test_class_counts.to_string())
    f.write("\n=== Random Forest Results ===\n")
    f.write(f"Timestamp: {TIMESTAMP}\n\n")
    f.write(f"Dropped correlated features: {to_drop}\n\n")
    best_rf = grid.best_estimator_.named_steps['classifier']
    f.write("Best RF params:\n")
    f.write(f"{str(grid.best_params_)}\n\n")
    f.write(f"Model saved to: {MODEL_PATH}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write(f"\nAccuracy: {acc:.4f}\n")
    f.write(f"Soft Accuracy (|pred - true| <= 1): {soft_acc:.4f}\n\n")

    # Feature importances
    importances = best_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    f.write("\nTop 10 Feature Importances:\n")
    for i in sorted_idx[:10]:
        f.write(f"{X_train.columns[i]}: {importances[i]:.4f}\n")

# --------------- PLOTS ----------------
sns.set(font_scale=1.5)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"conf_matrix_{TIMESTAMP}.png"))
plt.close()

end_time = datetime.now()
print(f"end time: {end_time.strftime("%Y%m%d_%H%M%S")}")
print(f"total time: {end_time - start_time}")
winsound.Beep(300, 1200)
