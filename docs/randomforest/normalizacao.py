import pandas as pd
from sklearn.preprocessing import LabelEncoder

def standardization(df):

    df['Z-Age'] = df['Age'].apply(lambda x: (x-df['Age'].mean())/df['Age'].std())
    df['N-Age'] = df['Age'].apply(lambda x: (x-df['Age'].min())/(df['Age'].max()-df['Age'].min()))
    df['Z-Na_to_K'] = df['Na_to_K'].apply(lambda x: (x-df['Na_to_K'].mean())/df['Na_to_K'].std())
    df['N-Na_to_K'] = df['Na_to_K'].apply(lambda x: (x-df['Na_to_K'].min())/(df['Na_to_K'].max()-df['Na_to_K'].min()))
    df = df[['Age', 'N-Age', 'Z-Age', 'Na_to_K', 'N-Na_to_K', 'Z-Na_to_K']].dropna()
    print(df.head(10).to_markdown())

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
df = df.sample(n=10, random_state=42)

# Preprocessing
df = preprocess(df)

standardization(df)