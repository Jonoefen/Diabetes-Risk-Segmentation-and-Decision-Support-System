import pandas as pd
from sklearn.model_selection import train_test_split

print(df.isnull().sum())

df = pd.read_csv("Diabetes_and_LifeStyle_Dataset_.csv")

num = df.select_dtypes(include = ['int64', 'float64']).columns

for column in num:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[column] = df[column].clip(lower, upper)

train_df, test_df = train_test_split(df, test_size = 0.2, stratify = df["target"], random_state = 42)

train_df.to_csv("train.csv", index = False)
test_df.to_csv("test.csv", index = False)