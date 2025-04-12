from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def logistic_regression(X_train, y_train, max_iter=100, C=1.0):
    model = LogisticRegression(max_iter=max_iter, C=C)
    model.fit(X_train, y_train)
    return model

def decision_tree(X_train, y_train, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    return model

def random_forest(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    return model

def naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def svm(X_train, y_train, kernel='rbf', C=1.0):
    model = SVC(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    return model

def knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model
