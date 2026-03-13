def create_features(df):
    df['Family_Size']=df['SibSp']+df['Parch']+1
    df['Is_alone']=df['Family_Size'].apply(lambda x:1 if x==1 else 0)
    return df