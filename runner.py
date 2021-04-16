import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, multilabel_confusion_matrix, accuracy_score
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from algorithms import neural_network, decision_tree, feature_selection


def generate_confusion_matrix(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    confusion_matrix = {}
    test_error = 0
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            test_error += 1
        if y_true[i] in confusion_matrix:
            if y_pred[i] in confusion_matrix[y_true[i]]:
                confusion_matrix[y_true[i]][y_pred[i]] += 1
            else:
                confusion_matrix[y_true[i]][y_pred[i]] = 1
        else:
            confusion_matrix[y_true[i]] = {}
            confusion_matrix[y_true[i]][y_pred[i]] = 1
    test_error /= len(y_true)
    return (test_error, confusion_matrix)


def generate_report_sklearn(classifier, X_train, y_train, X_test, y_test):
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (accuracy_score(y_test, y_pred), multilabel_confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    M = np.genfromtxt(
        "algorithms/data/impostor_phenomenon_data.csv",
        missing_values="",
        skip_header=1,
        delimiter=",",
        dtype=int,
    )
    X = M[:, 0:-1]
    y = M[:, -1]

    # X = feature_selection.select_best_features(5, X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2
    )

    classifier = AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(max_depth=2))
    test_acc, confusion_matrix = generate_report_sklearn(
        classifier, X_train, y_train, X_test, y_test
    )
    print(test_acc)
    print(confusion_matrix)