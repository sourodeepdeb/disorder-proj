import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

df_data = pd.read_csv(dir_path + "training_embedding.csv")

X_matrix = np.array(df_data['ada_embedding'].apply(lambda x: ast.literal_eval(x)).tolist())
y_matrix = np.array(df_data["target"].tolist())

mask = np.isin(y_matrix, [0, 1, 2, 3])
X_matrix = X_matrix[mask]
y_matrix = y_matrix[mask]


X_matrix, y_matrix = shuffle(X_matrix, y_matrix, random_state=42)

smote = SMOTE(random_state=42)
X_matrix, y_matrix = smote.fit_resample(X_matrix, y_matrix)

scaler = StandardScaler()
X_matrix_scaled = scaler.fit_transform(X_matrix)

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=1000,
    shuffle=True,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_matrix_scaled, y_matrix, cv=cv, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores):.4f}")

model.fit(X_matrix_scaled, y_matrix)

y_pred = model.predict(X_matrix_scaled)

accuracy = accuracy_score(y_matrix, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f}")

category_mapping = {0: "Suicide", 1: "Stress", 2: "OCD", 3: "Anxiety"}
for class_label in range(5): 
    class_mask = y_matrix == class_label
    class_accuracy = accuracy_score(y_matrix[class_mask], y_pred[class_mask])
    print(f"Accuracy for {category_mapping[class_label]}: {class_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_matrix, y_pred, target_names=list(category_mapping.values())))

cm = confusion_matrix(y_matrix, y_pred)
plt.figure(figsize=(12,10)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(category_mapping.values()), yticklabels=list(category_mapping.values()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
