import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
x = d[['N-Age', 'BP', 'Cholesterol', 'N-Na_to_K']]
y = d['Drug']

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
labels = classifier.classes_
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("<h3>Relatório de Classificação:</h3>")
print(report_df.to_html(classes="table table-bordered table-striped", border=0))

print("<h3>Matriz de Confusão:</h3>")
print(cm_df.to_html(classes="table table-bordered table-striped", border=0))

# Optional: Print feature importances
feature_importance = pd.DataFrame({
    'Feature': ['N-Age', 'BP', 'Cholesterol', 'N-Na_to_K'],
    'Importance': classifier.feature_importances_
})
print("<br>Feature Importances:")
print(feature_importance.sort_values(by='Importance', ascending=False).to_html())

tree.plot_tree(classifier)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())