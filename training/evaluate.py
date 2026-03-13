from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true, y_pred):

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="binary")
    metrics["recall"] = recall_score(y_true, y_pred, average="binary")
    metrics["f1_score"] = f1_score(y_true, y_pred, average="binary")
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred)

    return metrics


def evaluate(model, x_test, y_test):

    pred = model.predict(x_test)

    metrics = calculate_metrics(y_test, pred)

    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1 Score:", metrics["f1_score"])

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(metrics["classification_report"])

    return metrics