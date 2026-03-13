from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
import pandas as pd

def scale(x_train,x_test):

    x_train_scaled=pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns,index=x_train.index)
    x_test_scaled=pd.DataFrame(scaler.transform(x_test),columns=x_test.columns,index=x_test.index)

    return x_train_scaled,x_test_scaled