import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # <--- IMPORTANTE

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

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/alexandremartinelli11/machine-learning/refs/heads/main/data/kaggle/drug200.csv')

# Preprocessing
d = preprocess(df.copy())
d = standardization(d)

# Select features for clustering
X = d[['N-Age', 'BP', 'Cholesterol', 'N-Na_to_K', 'Drug']]

# Run K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(X)

# ----> PCA para reduzir a 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Também projetar os centróides no PCA
centroids_pca = pca.transform(kmeans.cluster_centers_)

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
