'''
A machine learning based credit card fraud
detection using the GA algorithm for feature
selection
'''

import pandas as pd, numpy as np, random, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lime.lime_tabular
from joblib import Parallel, delayed

def prepare_data(df):
    X = df.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
    y = df['isFraud']
    pipe = ColumnTransformer([
        ('num', Pipeline([('scl', MinMaxScaler())]), ['step', 'amount']),
        ('cat', Pipeline([('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), ['type'])
    ])
    X = pipe.fit_transform(X)
    return *train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), pipe

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
        print(f"🔄 Generation {g+1}")
        fits = Parallel(n_jobs=-1)(delayed(fitness)(Xtr, Xte, ytr, yte, chrom) for chrom in pop)
        best = np.argmax(fits)
        print(f"Best F1-Score: {fits[best]:.5f}")
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
    print("Accuracy :", accuracy_score(yte, preds))
    print("Precision:", precision_score(yte, preds, zero_division=0))
    print("Recall   :", recall_score(yte, preds, zero_division=0))
    print("F1-Score :", f1_score(yte, preds, zero_division=0))
    # joblib.dump(model, out)
    return model

def explain(Xte, model, fnames, selected, idx=0):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=Xte[:, selected],
        feature_names=[fnames[i] for i in selected],
        class_names=['Not Fraud', 'Fraud'],
        mode='classification'
    )
    explainer.explain_instance(Xte[idx, selected], model.predict_proba).show_in_notebook()

def main(df, pop=10, gen=10, mrate=0.1):
    Xtr, Xte, ytr, yte, pipe = prepare_data(df)
    population = generate_population(pop, Xtr.shape[1])
    best = evolve(Xtr, Xte, ytr, yte, population, gen, mrate)
    selected = [i for i, b in enumerate(best) if b]
    print(f"Selected Feature Indices: {selected}")
    model = train_rf(Xtr, Xte, ytr, yte, selected)
    cat_names = pipe.named_transformers_['cat']['ohe'].get_feature_names_out(['type'])
    feature_names = np.concatenate((['step', 'amount'], cat_names))
    explain(Xte, model, feature_names, selected)
