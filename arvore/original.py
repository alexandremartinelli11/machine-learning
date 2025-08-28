import pandas as pd

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/alexandremartinelli11/machine-learning/refs/heads/main/data/kaggle/drug200.csv')
df = df.sample(n=10, random_state=42)

# Display the first few rows of the dataset
print(df.to_markdown(index=False))