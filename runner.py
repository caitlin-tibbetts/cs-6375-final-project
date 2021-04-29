import numpy as np
from itertools import permutations, combinations

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    multilabel_confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from algorithms import k_nearest_neighbors, feature_selection, naive_bayes


def generate_report_sklearn(classifier, X_train, y_train, X_test, y_test):
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average="weighted"),
        multilabel_confusion_matrix(y_test, y_pred),
    )


if __name__ == "__main__":
    file_name = "algorithms/data/impostor_phenomenon_data_factsonly.csv"
    headers = np.genfromtxt(file_name, delimiter=",", dtype=str, max_rows=1)
    M = np.genfromtxt(
        file_name,
        missing_values="",
        skip_header=1,
        delimiter=",",
        dtype=int,
    )
    X = M[1:, 0:-1]
    y = M[1:, -1]

    print(f"Labels: {set(y)}")

    best_features, X = feature_selection.select_best_features(7, X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("AdaBoostClassifier")

    classifier = AdaBoostClassifier(
        n_estimators=20, base_estimator=DecisionTreeClassifier(max_depth=1)
    )
    test_acc, f1, confusion_matrix = generate_report_sklearn(
        classifier, X_train, y_train, X_test, y_test
    )
    print(test_acc)
    print(f1)
    print(confusion_matrix)

    """ Find best layer sizes
    layer_sizes = [2,3,4,5]
    all_possible_orientations = []
    for i in range(1,4):
        coms = combinations(layer_sizes, i)
        for l in coms:
            perms = permutations(l, i)
            for p in perms:
                all_possible_orientations.append(list(perms))
    all_possible_orientations = [item for sublist in all_possible_orientations for item in sublist]
        
    for layers in all_possible_orientations:
        print(layers)
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layers, random_state=1, max_iter=1000)
        test_acc, f1, confusion_matrix = generate_report_sklearn(
            classifier, X_train, y_train, X_test, y_test
        )
        print(test_acc)
        print(f1)
        #print(confusion_matrix)
    """

    print("MLPClassifier")

    classifier = MLPClassifier(hidden_layer_sizes=(4, 3), max_iter=1500)
    test_acc, f1, confusion_matrix = generate_report_sklearn(
        classifier, X_train, y_train, X_test, y_test
    )
    print(test_acc)
    print(f1)
    print(confusion_matrix)

    print("SVC")

    classifier = SVC(kernel="poly")
    test_acc, f1, confusion_matrix = generate_report_sklearn(
        classifier, X_train, y_train, X_test, y_test
    )
    print(test_acc)
    print(f1)
    print(confusion_matrix)

    """ Find best value for k
    for n in range(1,X.shape[0]):
        print(n)
        y_pred = [
            k_nearest_neighbors.predict(X_train, y_train, n, X_test[i])
            for i in range(X_test.shape[0])
        ]
        test_acc, f1, confusion_matrix = (
            accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred, average="weighted"),
            multilabel_confusion_matrix(y_test, y_pred),
        )
        print(test_acc)
    """

    print("K-Nearest-Neighbors")
    
    y_pred = [
        k_nearest_neighbors.predict(X_train, y_train, 40, X_test[i])
        for i in range(X_test.shape[0])
    ]
    test_acc, f1, confusion_matrix = (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average="weighted"),
        multilabel_confusion_matrix(y_test, y_pred),
    )
    print(test_acc)
    print(f1)
    print(confusion_matrix)
    
    print("Naive Bayes Classifier")

    model = naive_bayes.fit(X_train, y_train)
    
    y_pred = [
        naive_bayes.predict(model, X_test[i])
        for i in range(X_test.shape[0])
    ]
    
    accuracy, f1, confusion_matrix = (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average="weighted"),
        multilabel_confusion_matrix(y_test, y_pred),
    )
    print("Accuracy: ", accuracy)
    print("f1_score: ", f1)
    print("Confusion Matrix: ", confusion_matrix)
