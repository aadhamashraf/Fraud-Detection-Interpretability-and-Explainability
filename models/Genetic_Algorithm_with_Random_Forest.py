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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lime.lime_tabular
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

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
    preds = model.predict(Xte[:, features])
    print("Classification Report:")
    print(classification_report(yte, preds, zero_division=0))
    visualize_metrics(yte, preds)
    return model

def visualize_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Accuracy': accuracy}
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, metric_values, color=['green', 'orange', 'blue', 'red'])
    plt.title('Classification Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for i, value in enumerate(metric_values):
        plt.text(i, value + 0.02, f'{value:.2f}', ha='center')
    plt.show()

def explain(Xte, model, fnames, selected, idx=0):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=Xte[:, selected],
        feature_names=[fnames[i] for i in selected],
        class_names=['Not Fraud', 'Fraud'],
        mode='classification'
    )
    explainer.explain_instance(Xte[idx, selected], model.predict_proba).show_in_notebook()

def main(df, pop=10, gen=10, mrate=0.1):
    X_train_resampled, X_test, y_train_resampled, y_test, pipe = prepare_data(df)
    pop = generate_population(pop=10, dim=X_train_resampled.shape[1])
    best_gene = evolve(X_train_resampled, X_test, y_train_resampled, y_test, pop, gen=10, mrate=0.1)
    selected_features = [i for i, bit in enumerate(best_gene) if bit]
    train_rf(X_train_resampled, X_test, y_train_resampled, y_test, selected_features)
