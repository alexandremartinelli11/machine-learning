import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import cycle

plt.figure(figsize=(12, 10))

def standardization(df):

    df['Z-Age'] = df['Age'].apply(lambda x: (x-df['Age'].mean())/df['Age'].std())
    df['N-Age'] = df['Age'].apply(lambda x: (x-df['Age'].min())/(df['Age'].max()-df['Age'].min()))
    df['Z-Na_to_K'] = df['Na_to_K'].apply(lambda x: (x-df['Na_to_K'].mean())/df['Na_to_K'].std())
    df['N-Na_to_K'] = df['Na_to_K'].apply(lambda x: (x-df['Na_to_K'].min())/(df['Na_to_K'].max()-df['Na_to_K'].min()))
    features = ['N-Age', 'Sex', 'BP', 'Cholesterol', 'N-Na_to_K', 'Drug']
    return df[features]

def preprocess(df):
    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Sex'].fillna(df['Sex'].mode()[0], inplace=True)
    df['BP'].fillna(df['BP'].mode()[0], inplace=True)
    df['Cholesterol'].fillna(df['Cholesterol'].mode()[0], inplace=True)
    df['Na_to_K'].fillna(df['Na_to_K'].median(), inplace=True)
    df['Drug'].fillna(df['Drug'].mode()[0], inplace=True)

   # Convert categorical variables
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['BP'] = label_encoder.fit_transform(df['BP'])
    df['Cholesterol'] = label_encoder.fit_transform(df['Cholesterol'])
    df['Drug'] = label_encoder.fit_transform(df['Drug'])

    # Select features
    features = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
    return df[features]

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/alexandremartinelli11/machine-learning/refs/heads/main/data/kaggle/drug200.csv')

# Preprocessing
d = preprocess(df.copy())
d = standardization(d)


# Generate synthetic dataset
X = d[['N-Age', 'BP', 'Cholesterol', 'N-Na_to_K']]
y = d['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Obter as probabilidades de previsão para todas as classes
y_scores = knn.predict_proba(X_test)

# Mapear classes e cores
classes = np.unique(y_test)
n_classes = len(classes)
colors = cycle(['blue', 'red', 'green', 'orange', 'yellow'])
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

plt.figure(figsize=(10, 8))

# Abordagem One-vs-Rest
for i, color in zip(classes, colors):
    # Trata a classe `i` como positiva e o restante como negativo
    y_test_bin = (y_test == i).astype(int)
    
    # Obter as probabilidades para a classe `i`
    y_scores_class = y_scores[:, i]

    fpr, tpr, _ = roc_curve(y_test_bin, y_scores_class)
    roc_auc = auc(fpr, tpr)
    
    fpr_dict[i] = fpr
    tpr_dict[i] = tpr
    roc_auc_dict[i] = roc_auc
    
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC da Classe {i} (área = {roc_auc:.2f})')

# Plotar a linha base do classificador aleatório
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Classificador Aleatório')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curvas ROC Multiclasse (One-vs-Rest)')
plt.legend(loc='lower right')
plt.grid(True)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

