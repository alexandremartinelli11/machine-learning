import pandas as pd

# Load the dataset
df = pd.read_csv('data/kaggle/drug200.csv')
df = df.sample(n=10, random_state=42)

# Display the first few rows of the dataset
print(df.to_markdown(index=False))