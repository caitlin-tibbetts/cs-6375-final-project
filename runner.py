import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from algorithms import neural_network, decision_tree, feature_selection


def generate_report_sklearn(classifier, X_train, y_train, X_test, y_test):
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), multilabel_confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    file_name = "algorithms/data/impostor_phenomenon_data_factsonly.csv"
    headers = np.genfromtxt(file_name, delimiter=',', dtype=str, max_rows=1)
    M = np.genfromtxt(
        file_name,
        missing_values="",
        skip_header=1,
        delimiter=",",
        dtype=int,
    )
    X = M[1:, 0:-1]
    y = M[1:, -1]

    best_features, X = feature_selection.select_best_features(7, X, y)
    print(headers[best_features])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2
    )

    classifier = AdaBoostClassifier(n_estimators=20, base_estimator=DecisionTreeClassifier(max_depth=1))
    test_acc, f1, confusion_matrix = generate_report_sklearn(
        classifier, X_train, y_train, X_test, y_test
    )
    print(test_acc)
    print(f1)
    print(confusion_matrix)