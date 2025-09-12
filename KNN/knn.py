import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


r = permutation_importance(
    knn,                  
    X_test,               
    y_test,               
    n_repeats=30,         
    random_state=42,
    scoring='accuracy'    
)


feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': r.importances_mean,
    'Std': r.importances_std
})

# ordenar e mostrar (HTML igual ao seu exemplo)
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("<br>Feature Importances (Permutation):")
print(feature_importance.to_html(index=False))


# Escalar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduzir para 2 dimensões (apenas para visualização)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Treinar KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)


# Visualize decision boundary
h = 0.02  # Step size in mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, style=y, palette="deep", s=100)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Decision Boundary (k=3)")

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
