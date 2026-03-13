from src.load_data import load_data
from src.preprocess import preprocessing
from src.encoding import encode
from src.scaling import scale
from src.split_data import split_data

from sklearn.linear_model import LogisticRegression


def run_training(path):

    # load data
    df = load_data(path)

    # preprocessing
    df = preprocessing(df)

    # split
    x_train, x_test, y_train, y_test = split_data(df)

    # encoding
    x_train_encode, x_test_encode = encode(x_train, x_test)

    # scaling
    x_train_scaled, x_test_scaled = scale(x_train_encode, x_test_encode)

    # train model
    model = LogisticRegression()

    model.fit(x_train_scaled, y_train)

    return model, x_test_scaled, y_test
