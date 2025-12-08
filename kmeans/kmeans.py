import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA  


plt.figure(figsize=(12, 10))

def standardization(df):
    df['Z-Age'] = df['Age'].apply(lambda x: (x - df['Age'].mean()) / df['Age'].std())
    df['N-Age'] = df['Age'].apply(lambda x: (x - df['Age'].min()) / (df['Age'].max() - df['Age'].min()))
    df['Z-Na_to_K'] = df['Na_to_K'].apply(lambda x: (x - df['Na_to_K'].mean()) / df['Na_to_K'].std())
    df['N-Na_to_K'] = df['Na_to_K'].apply(lambda x: (x - df['Na_to_K'].min()) / (df['Na_to_K'].max() - df['Na_to_K'].min()))
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

    features = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
    return df[features]

label_encoder = LabelEncoder()
# Load dataset
df = pd.read_csv('data/kaggle/drug200.csv')

# Preprocessing
d = preprocess(df.copy())
d = standardization(d)

X = d[['N-Age', 'BP', 'Cholesterol', 'N-Na_to_K']]
y = d['Drug']

#Separar em teste e validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reduzir para 2 dimensões com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Treinar KMeans
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Mapear clusters para classes reais por voto majoritário
cluster_map = {}
for c in np.unique(labels):
    mask = labels == c
    majority_class = mode(y_train[mask], keepdims=False)[0]
    cluster_map[c] = majority_class

# Reatribuir clusters como classes previstas
y_pred = np.array([cluster_map[c] for c in labels])

# Calcular acurácia e matriz de confusão
acc = accuracy_score(y_train, y_pred)
cm = confusion_matrix(y_train, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=[f"Classe Real {cls}" for cls in np.unique(y_train)],
    columns=[f"Classe Pred {cls}" for cls in np.unique(y_train)]
)

print(f"Acurácia: {acc:.2f}%")
print("<br>Matriz de Confusão:")
print(cm_df.to_html())

# Também projetar os centróides no PCA
centroids_pca = kmeans.cluster_centers_
                        

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='red', marker='*', s=200, label='Centroids')
plt.title('K-Means Clustering (PCA 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Save plot to buffer
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
