from statistics import mode

def predict(X_train, y_train, k, example):
    nearest_neighbors = []
    for i in range(X_train.shape[0]):
        distance = sum([abs(X_train[i,f]-example[f]) for f in range(X_train.shape[1])])
        nearest_neighbors.append((distance, y_train[i]))
        nearest_neighbors = sorted(nearest_neighbors)
        if len(nearest_neighbors) == k+1:
            nearest_neighbors = nearest_neighbors[:-1]
    return mode([y for d,y in nearest_neighbors])
        