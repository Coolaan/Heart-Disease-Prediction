import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
uploaded = files.upload()
# Load dataset
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Filter individuals above the age of 50
data_filtered = data[data['age'] > 50]

# Split features and target variable
X = data_filtered.drop("DEATH_EVENT", axis=1)
y = data_filtered["DEATH_EVENT"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define pipeline for supervised learning
supervised_pipeline = Pipeline([
    ('select_features', SelectKBest(f_classif, k=5)),
    ('reduce_dim', PCA(n_components=5)),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=None))
])

# Fit supervised model
supervised_pipeline.fit(X_train_scaled, y_train)

# Predictions on test set
y_pred = supervised_pipeline.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Supervised Learning Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Additional preprocessing
robust_scaler = StandardScaler()
X_train_scaled_robust = robust_scaler.fit_transform(X_train)
X_test_scaled_robust = robust_scaler.transform(X_test)

# Define pipelines with SMOTE
pipelines_with_smote = [
    ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=None))
    ]),
    ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression())
    ])
]

# Initialize lists to store model names and their corresponding F1 scores
model_names = []
f1_scores = []

# Iterate over each pipeline with SMOTE
for pipeline in pipelines_with_smote:
    # Fit pipeline
    pipeline.fit(X_train_scaled, y_train)

    # Predictions on test set
    y_pred = pipeline.predict(X_test_scaled)

    # Evaluate F1 score
    f1 = f1_score(y_test, y_pred)
    
    # Append model name and F1 score to lists
    model_names.append(pipeline.steps[-1][1]._class.name_)
    f1_scores.append(f1)

# Plot model comparison chart
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=f1_scores, palette="viridis")
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('Model Comparison with SMOTE')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calculate correlation matrix
corr_matrix = X_train.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()
