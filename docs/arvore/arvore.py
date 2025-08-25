import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

    # Select features
    features = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
    return df[features]

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/alexandremartinelli11/machine-learning/refs/heads/main/data/kaggle/drug200.csv')

# Preprocessing
d = preprocess(df.copy())
d = standardization(d)

plt.figure(figsize=(12, 10))

# Carregar o conjunto de dados
x = d[['N-Age', 'Sex', 'BP', 'Cholesterol', 'N-Na_to_K']]
y = d['Drug']

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Optional: Print feature importances
feature_importance = pd.DataFrame({
    'Feature': ['N-Age', 'Sex', 'BP', 'Cholesterol', 'N-Na_to_K'],
    'Importance': classifier.feature_importances_
})
print("<br>Feature Importances:")
print(feature_importance.sort_values(by='Importance', ascending=False).to_html())

tree.plot_tree(classifier)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())