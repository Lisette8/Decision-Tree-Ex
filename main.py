import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('dataset_decision_tree.csv')
dataset.info()
print(dataset.describe())

# Matrice de corrélation
rcParams['figure.figsize'] = 12, 10
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns, rotation=90)
plt.colorbar()
plt.show()

# Histogrammes
dataset.hist()
plt.show()

# Répartition des classes
rcParams['figure.figsize'] = 8, 6
counts = dataset['approved'].value_counts().sort_index()
plt.bar(counts.index, counts.values, color=['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.show()

# Séparation features / cible
X = dataset.drop('approved', axis=1)
y = dataset['approved']  # 1 = prêt accepté, 0 = refusé

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (optionnel pour un arbre), après le split
columns_to_scale = ['income', 'credit_score', 'loan_amount', 'years_employed']
standardScaler = StandardScaler()
X_train[columns_to_scale] = standardScaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = standardScaler.transform(X_test[columns_to_scale])

# Modèle
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Evaluation
print("\nEVALUATION:\n")
# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



# Decision tree visualization
plt.figure(figsize=(20,10))
tree.plot_tree(
    clf,
    filled=True,
    feature_names=X.columns,
    class_names=['Refusé', 'Accepté'],
    rounded=True
)
plt.show()
