from training.train_models import run_training
from training.evaluate import evaluate

path=r"C:\Users\Srimanth\Desktop\Titanic-Dataset.csv"
model, x_test, y_test = run_training(path)

metrics = evaluate(model, x_test, y_test)