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
from sklearn.inspection import PartialDependenceDisplay , permutation_importance
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
    y_pred = model.predict(Xte)
    y_probs = model.predict_proba(Xte)[:, 1]
    visualize_metrics(yte, y_pred, y_probs, model, [f"feat_{i}" for i in features])
    return model


def train_nb(Xtr, Xte, ytr, yte, features):
    model = GaussianNB()
    model.fit(Xtr[features], ytr)
    y_pred = model.predict(Xte[features])
    y_probs = model.predict_proba(Xte[features])[:, 1]
    visualize_metrics(yte, y_pred, y_probs, model, features)
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

def explain_lime(Xte, model, fnames, selected, idx=0):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=Xte[:, selected],
        feature_names=[fnames[i] for i in selected],
        class_names=['Not Fraud', 'Fraud'],
        mode='classification'
    )
    explainer.explain_instance(Xte[idx, selected], model.predict_proba).show_in_notebook()

def explain_pdp_ice(model, X_test, selected, feature_names, pipe):
    selected_features = [feature_names[i] for i in selected]    
    fig, ax = plt.subplots(nrows=len(selected_features), figsize=(8, 3 * len(selected_features)))
    for i, feat_idx in enumerate(selected):
        PartialDependenceDisplay.from_estimator(
            model, 
            X_test[:, selected], 
            [i],  
            feature_names=selected_features,
            kind='both',  
            ax=ax[i] if len(selected_features) > 1 else ax
        )
    plt.tight_layout()
    plt.show()

def explain_permutation_importance(model, X_test, y_test, selected, feature_names):
    result = permutation_importance(model, X_test[:, selected], y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    sorted_feats = [feature_names[selected[i]] for i in sorted_idx]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=result.importances_mean[sorted_idx], y=sorted_feats)
    plt.title('Permutation Feature Importance')
    plt.xlabel('Mean Importance Decrease')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def main(df, pop=10, gen=10, mrate=0.1):
    X_train_resampled, X_test, y_train_resampled, y_test, pipe = prepare_data(df)
    feat_names = pipe.get_feature_names_out()
    
    # Genetic Algorithm: Select features using Random Forest internally
    pop = generate_population(pop=pop, dim=X_train_resampled.shape[1])
    best_gene = evolve(X_train_resampled, X_test, y_train_resampled, y_test, pop, gen=gen, mrate=mrate)
    selected_features = [i for i, bit in enumerate(best_gene) if bit] # ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'newbalanceDest', 'type_PAYMENT', 'type_TRANSFER']
    
    # train_rf(X_train_resampled, X_test, y_train_resampled, y_test, selected_features)
    model, selected_features = train_nb(X_train_resampled, X_test, y_train_resampled, y_test, selected_features) 
    selected_indices = [X_test.columns.get_loc(col) for col in selected_features]

    ## **Explainability  Section**
    explain_lime(X_test, model, selected_features, selected_indices, idx=0)
    explain_pdp_ice(model, X_test, selected_features, feat_names, pipe)
    explain_permutation_importance(model, X_test, y_test, selected_features, feat_names)
