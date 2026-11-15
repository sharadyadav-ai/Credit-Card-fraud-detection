import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# 1. Load the dataset (Make sure creditcard.csv is in the same folder)
print("Loading dataset...")
try:
  df = pd.read_csv("creditcard.csv")
except FileNotFoundError:
  print(
      "Error: creditcard.csv not found. Please download it from Kaggle and place"
      " it in the project root."
  )
  exit()

# 2. Separate features (X) and target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Train/Test Split
# stratify=y is crucial for extreme imbalance to ensure train/test sets have the same ratio of fraud.
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build the ML Pipeline
# RobustScaler is used instead of StandardScaler because 'Amount' and 'Time' have extreme outliers.
print("Building and training the pipeline...")
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    (
        "classifier",
        LogisticRegression(class_weight="balanced", max_iter=1000),
    ),
])

# 5. Train the model
pipeline.fit(X_train, y_train)

# 6. Evaluate the model
print("\n--- Model Evaluation ---")
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Print standard classification metrics
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# Precision-Recall AUC is the standard metric for severely imbalanced datasets
pr_auc = average_precision_score(y_test, y_prob)
print(f"Precision-Recall AUC (PR-AUC): {pr_auc:.4f}\n")

# 7. Save the full pipeline to disk (this bundles scaler + model together)
model_filename = "fraud_pipeline.pkl"
joblib.dump(pipeline, model_filename)
print(f"Success! Model pipeline saved as {model_filename}")