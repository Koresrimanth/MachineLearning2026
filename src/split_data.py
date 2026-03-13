
from sklearn.model_selection import train_test_split
from utils.drop_columns import drop_unused_columns

def split_data(df):
    
    cols_to_drop=["PassengerId",
            "Name",
            "Ticket",
            "Cabin"]
    
    df=drop_unused_columns(df,cols_to_drop)

    x=df.drop(columns=['Survived'])

    y=df['Survived']

    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

    return x_train, x_test, y_train, y_test
