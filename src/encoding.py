from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


def encode(x_train,x_test):

    deck_order=['M', 'G', 'F', 'E', 'D', 'C', 'B', 'A', 'T']

    preprocessor=ColumnTransformer(
        transformers=[
            ('ohe',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore')['Sex','Embarked']),
            ('ordinal',OrdinalEncoder(categories=[deck_order],
            handle_unknown='use_encoded_value',
            unknown_value=-1)['Deck'])
        ],
        remainder='passthrough'
    )
    x_train_transformed=x_train.fit_transform(x_train)
    x_test_transformed=x_test.transform(x_test)

    x_cols=preprocessor.get_feature_names_out()
    
    x_train_encode=pd.DataFrame(x_train_transformed,columns=x_cols,index=x_train.index)
    x_test_encode=pd.DataFrame(x_test_transformed,columns=x_cols,index=x_test.index)

    return x_train_encode,x_test_encode