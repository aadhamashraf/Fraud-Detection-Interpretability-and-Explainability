'''
A machine learning based credit card fraud
detection using the GA algorithm for feature
selection
'''

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lime.lime_tabular
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

def prepare_data(df):
    X = df.drop(columns=['isFraud', 'isFlaggedFraud'])
    y = df['isFraud']
    pipe = ColumnTransformer([
        ('num', Pipeline([('scl', MinMaxScaler())]), ['step', 'amount']),
        ('cat', Pipeline([('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), ['type'])
    ])
    X = pipe.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test, pipe

def generate_population(pop, dim):
    return [np.random.choice([0, 1], size=dim).tolist() for _ in range(pop)]

def fitness(Xtr, Xte, ytr, yte, mask):
    idx = [i for i, b in enumerate(mask) if b]
    if not idx: return 0
    sample = np.random.choice(len(ytr), size=max(5000, int(0.1 * len(ytr))), replace=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(Xtr[sample][:, idx], ytr.iloc[sample])
    return f1_score(yte, model.predict(Xte[:, idx]), zero_division=0)

def evolve(Xtr, Xte, ytr, yte, pop, gen, mrate):
    best_score, best_gene = 0, None
    for g in range(gen):
        fits = Parallel(n_jobs=-1)(delayed(fitness)(Xtr, Xte, ytr, yte, chrom) for chrom in pop)
        best = np.argmax(fits)
        if fits[best] > best_score: best_score, best_gene = fits[best], pop[best]
        parents = [pop[i] for i in np.argsort(fits)[-4:]]
        pop = [mutate(*crossover(*random.sample(parents, 2)), mrate) for _ in range(len(pop)//2) * 2]
        pop = [item for pair in pop for item in (pair if isinstance(pair, tuple) else (pair,))]
    return best_gene

def crossover(a, b):
    p = random.randint(1, len(a) - 1)
    return a[:p] + b[p:], b[:p] + a[p:]

def mutate(chrom, rate):
    return [bit if random.random() > rate else 1 - bit for bit in chrom]

def train_rf(Xtr, Xte, ytr, yte, features, out='best_rf_model.pkl'):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(Xtr[:, features], ytr)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    visualize_metrics(y_test, y_pred, y_probs, clf, X.columns)
    return model


def train_nb(Xtr, Xte, ytr, yte, features):
    model = GaussianNB()
    model.fit(Xtr[:, features], ytr)
    y_pred = model.predict(Xte[:, features])
    y_probs = model.predict_proba(Xte[:, features])[:, 1]
    visualize_metrics(yte, y_pred, y_probs, model, [f"feat_{i}" for i in features])
    return model, features


def visualize_metrics(y_true, y_pred, y_probs, model, feature_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:")
    print(report_df)

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def explain(Xte, model, fnames, selected, idx=0):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=Xte[:, selected],
        feature_names=[fnames[i] for i in selected],
        class_names=['Not Fraud', 'Fraud'],
        mode='classification'
    )
    explainer.explain_instance(Xte[idx, selected], model.predict_proba).show_in_notebook()
    explanation.show_in_notebook()

def main(df, pop=10, gen=10, mrate=0.1):
    X_train_resampled, X_test, y_train_resampled, y_test, pipe = prepare_data(df)
    
    # Genetic Algorithm: Select features using Random Forest internally
    pop = generate_population(pop=pop, dim=X_train_resampled.shape[1])
    best_gene = evolve(X_train_resampled, X_test, y_train_resampled, y_test, pop, gen=gen, mrate=mrate)
    selected_features = [i for i, bit in enumerate(best_gene) if bit] # ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'newbalanceDest', 'type_PAYMENT', 'type_TRANSFER']
    # train_rf(X_train_resampled, X_test, y_train_resampled, y_test, selected_features)

    model, selected_features = train_nb(X_train_resampled, X_test, y_train_resampled, y_test, selected_features) 
    explain(X_test, model, [f"feat_{i}" for i in range(X_test.shape[1])], selected_features, idx=0)

