import pandas as pd
import numpy as np


def preprocessing(df):
    #handle the misssing values
    df['Age']=df["Age"].fillna(df['Age'].mean())
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Deck']=df['Cabin'].apply(lambda x:x[0] if pd.notnull(x) else 'M')

    #create features
    df['Family_Size']=df['SibSp']+df['Parch']+1
    df['Is_alone']=df['Family_Size'].apply(lambda x:1 if x==1 else 0)

    #fixing the skwewness
    df['Fare']=df['Fare'].apply(np.log1p)

    return df







