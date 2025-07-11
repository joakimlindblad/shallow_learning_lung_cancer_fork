import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from tqdm import tqdm

# Toggle permutation importances
PERMUTE_FEATURES = True

def get_feature_importance(model, columns):
    if hasattr(model, 'coef_'):
        return pd.DataFrame({'feature': columns, 'importance': model.coef_[0]})
    elif hasattr(model, 'feature_importances_'):
        return pd.DataFrame({'feature': columns, 'importance': model.feature_importances_})
    else:
        return pd.DataFrame({'feature': columns, 'importance': [np.nan]*len(columns)})

def plot_feature_importances(importances, feature_names, title, filename):
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class RandomClassifier:
    def fit(self, X, y):
        self.label_proportions = y.value_counts(normalize=True)
    def predict(self, X):
        return np.random.choice(self.label_proportions.index, size=len(X), p=self.label_proportions.values)
    def predict_proba(self, X):
        return np.tile(self.label_proportions.values, (len(X), 1))

def init_rf():
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt']
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    return GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def init_logreg():
    param_grid = {
        'C': np.logspace(-4, 4, 10),#[0.1, 1, 10],#(-4, 4, 10),  # Try a wide range of regularization strengths
        'penalty': ['l1', 'l2'],      # Test both L1 and L2 regularization
        'solver': ['liblinear', 'saga'],  # Solvers that support L1 and L2 penalties
        'max_iter': [10000]   # Ensure enough iterations for convergence
    }
    log_reg = LogisticRegression(random_state=42)
    return GridSearchCV(log_reg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def init_svm():
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
    }
    svm = SVC(probability=True, random_state=42)
    return GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

def init_knn():
    param_grid = {'n_neighbors': [5, 10, 20]}
    knn = KNeighborsClassifier()
    return GridSearchCV(knn, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

model_registry = {
    "RF": init_rf,
    "LogReg": init_logreg,
    "SVM": init_svm,
    "KNN": init_knn,
    #"Random": lambda: RandomClassifier()
}

# ---- Main ----

df = pd.read_csv("patient_densities_morphologies.csv")
X = df.drop(columns=["ID", 'time', 'event', 'label'])
y = df[["ID", "label"]]

column_names = X.columns.tolist()
clinical_parameters = column_names[:6]
morphologies = column_names[6:9]
densities = column_names[9:]
part_names = {"clinical parameters": clinical_parameters, "morphologies": morphologies, "densities": densities}

name_combination_list = [
    ["clinical parameters","morphologies","densities"],
    ["clinical parameters","morphologies"],
    ["clinical parameters","densities"],
    ["morphologies","densities"],
    ["clinical parameters"],
    ["morphologies"],
    ["densities"]
]

split_folder = "/data2/love/multiplex_cancer_cohorts/patient_and_samples_data/lung_cancer_BOMI2_dataset/binary_survival_prediction/100foldcrossvalrepeat/"
num_splits = len([x for x in os.listdir(split_folder) if "test" in x])

os.makedirs("plots", exist_ok=True)
os.makedirs("preds", exist_ok=True)

results_dict_titles = [
    "Experiment", "Model",
    "Train accuracy mean", "Train accuracy std", "Train AUC mean", "Train AUC std",
    "Test accuracy mean", "Test accuracy std", "Test accuracy sem",
    "Test AUC mean", "Test AUC std", "Test AUC sem",
    "Test sensitivity", "Test specificity", "Test NPV", "Test PPV"
]
results_dict = {t: [] for t in results_dict_titles}

experiment_results_acc = {}
experiment_results_auc = {}

# For p-value stats
all_acc = {}
all_auc = {}

for name_comb in name_combination_list:
    experiment_name = "_".join(name_comb)
    feature_list = sum([part_names[part] for part in name_comb], [])
    print(f"=== Experiment: {experiment_name} | Features: {feature_list}")

    X_experiment = X[feature_list]
    for model_name, model_fn in model_registry.items():
        # Storage for per-split
        test_accs, test_aucs, test_sens, test_spec, test_npv, test_ppv = [], [], [], [], [], []
        train_accs, train_aucs = [], []
        feature_importances = []
        permutation_importances = []
        preds_by_id = {ID: [] for ID in df["ID"]}
        logits_by_id = {ID: [] for ID in df["ID"]}
        preds_logits = []

        for split in tqdm(range(num_splits), desc=f"{experiment_name}-{model_name}", leave=False):
            train_ids = pd.read_csv(os.path.join(split_folder, f"split_{split}_train_val.csv"))["ID"]
            test_ids = pd.read_csv(os.path.join(split_folder, f"split_{split}_test.csv"))["ID"]
            train_mask = df["ID"].isin(train_ids)
            test_mask = df["ID"].isin(test_ids)
            X_train, X_test = X_experiment[train_mask], X_experiment[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = model_fn()
            # Fit
            if model_name == "Random":
                model.fit(X_train_scaled, y_train["label"])
                preds = model.predict(X_test_scaled)
                probas = model.predict_proba(X_test_scaled)[:, 1]
                preds_train = model.predict(X_train_scaled)
                probas_train = model.predict_proba(X_train_scaled)[:, 1]
                best_model = model
            else:
                model.fit(X_train_scaled, y_train["label"])
                best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
                preds = best_model.predict(X_test_scaled)
                probas = best_model.predict_proba(X_test_scaled)[:, 1]
                preds_train = best_model.predict(X_train_scaled)
                probas_train = best_model.predict_proba(X_train_scaled)[:, 1]

            # Metrics
            acc = accuracy_score(y_test["label"], preds)
            auc = roc_auc_score(y_test["label"], probas)
            acc_train = accuracy_score(y_train["label"], preds_train)
            auc_train = roc_auc_score(y_train["label"], probas_train)
            test_accs.append(acc)
            test_aucs.append(auc)
            train_accs.append(acc_train)
            train_aucs.append(auc_train)

            cm = confusion_matrix(y_test["label"], preds)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
            sensitivity = recall_score(y_test["label"], preds) if (tp+fn) else 0
            specificity = tn / (tn+fp) if (tn+fp) else 0
            ppv = precision_score(y_test["label"], preds) if (tp+fp) else 0
            npv = tn / (tn+fn) if (tn+fn) else 0
            test_sens.append(sensitivity)
            test_spec.append(specificity)
            test_npv.append(npv)
            test_ppv.append(ppv)

            # Store per-patient results
            test_ids_vals = df[test_mask]["ID"].values
            preds_logits.append(pd.DataFrame({
                "ID": test_ids_vals,
                "split": split,
                "prediction": preds,
                "logit": probas
            }))
            for ID, pred, logit in zip(test_ids_vals, preds, probas):
                preds_by_id[ID].append(pred)
                logits_by_id[ID].append(logit)

            # Feature importances
            if model_name in ["LogReg"]:
                fi = get_feature_importance(best_model, feature_list)
                feature_importances.append(fi)
                if PERMUTE_FEATURES:
                    pi = permutation_importance(best_model, X_test_scaled, y_test["label"], n_repeats=10, random_state=42, n_jobs=-1)
                    permutation_importances.append(pi.importances_mean)

        # Aggregate results for results_dict
        results_dict["Experiment"].append(experiment_name)
        results_dict["Model"].append(model_name)
        results_dict["Train accuracy mean"].append(np.mean(train_accs))
        results_dict["Train accuracy std"].append(np.std(train_accs))
        results_dict["Train AUC mean"].append(np.mean(train_aucs))
        results_dict["Train AUC std"].append(np.std(train_aucs))
        results_dict["Test accuracy mean"].append(np.mean(test_accs))
        results_dict["Test accuracy std"].append(np.std(test_accs))
        results_dict["Test accuracy sem"].append(np.std(test_accs)/np.sqrt(len(test_accs)))
        results_dict["Test AUC mean"].append(np.mean(test_aucs))
        results_dict["Test AUC std"].append(np.std(test_aucs))
        results_dict["Test AUC sem"].append(np.std(test_aucs)/np.sqrt(len(test_aucs)))
        results_dict["Test sensitivity"].append(np.mean(test_sens))
        results_dict["Test specificity"].append(np.mean(test_spec))
        results_dict["Test NPV"].append(np.mean(test_npv))
        results_dict["Test PPV"].append(np.mean(test_ppv))

        # Store for stats
        all_acc[(experiment_name, model_name)] = test_accs
        all_auc[(experiment_name, model_name)] = test_aucs

        # Per-ID, split predictions/logits
        predictions_df = pd.DataFrame(preds_by_id)
        logits_df = pd.DataFrame(logits_by_id)
        preds_logits_cat = pd.concat(preds_logits, ignore_index=True)
        preds_logits_grouped = preds_logits_cat.groupby("ID").agg({
            "prediction": "mean",
            "logit": "mean"
        }).rename(columns={"prediction": "mean_prediction", "logit": "mean_logit"})
        preds_logits_grouped.to_csv(f"preds/{experiment_name}_{model_name}_preds_logits.csv", index=False)
        predictions_df.to_csv(f"preds/{experiment_name}_{model_name}_predictions.csv", index=False)
        logits_df.to_csv(f"preds/{experiment_name}_{model_name}_logits.csv", index=False)

        # Feature importances and plots
        if feature_importances:
            fi_cat = pd.concat(feature_importances)
            avg_fi = fi_cat.groupby('feature')['importance'].mean().reset_index()
            avg_fi = avg_fi.sort_values(by='importance', ascending=False)
            avg_fi.to_csv(f"plots/{experiment_name}_{model_name}_feature_importance.csv", index=False)
            plot_feature_importances(avg_fi['importance'].values, avg_fi['feature'].values,
                f"Feature Importances: {experiment_name} {model_name}",
                f"plots/{experiment_name}_{model_name}_feature_importance.png")
        # Permutation importance (if enabled)
        if PERMUTE_FEATURES and permutation_importances:
            avg_perm = np.mean(permutation_importances, axis=0)
            np.save(f"plots/{experiment_name}_{model_name}_perm_importance.npy", avg_perm)
            # Optionally plot
            plot_feature_importances(avg_perm, feature_list,
                f"Permutation Importances: {experiment_name} {model_name}",
                f"plots/{experiment_name}_{model_name}_perm_importance.png")

    # Save for baseline comparison (for stats)
    experiment_results_acc[experiment_name] = all_acc.get((experiment_name, "LogReg"), [])
    experiment_results_auc[experiment_name] = all_auc.get((experiment_name, "LogReg"), [])

# Save results to CSV
results = pd.DataFrame(results_dict)
results.to_csv("results_shallow_learning2_full.csv", index=False)

# Paired t-test vs clinical baseline (only for accuracy/AUC)
p_value_stats = {"experiment": [], "model": [], "Accuracy": [], "AUC": []}

for key in all_acc:
    experiment_name, model_name = key
    clinical_acc = all_acc.get(("clinical parameters", model_name))
    clinical_auc = all_auc.get(("clinical parameters", model_name))
    if experiment_name == "clinical parameters":# and model_name == "LogReg":
        continue
    results_acc = all_acc[key]
    results_auc = all_auc[key]
    if clinical_acc is not None and len(results_acc) == len(clinical_acc):
        _, p_value_acc = ttest_rel(results_acc, clinical_acc)
        _, p_value_auc = ttest_rel(results_auc, clinical_auc)
        p_value_stats["experiment"].append(experiment_name)
        p_value_stats["model"].append(model_name)
        p_value_stats["Accuracy"].append(p_value_acc)
        p_value_stats["AUC"].append(p_value_auc)
pd.DataFrame(p_value_stats).to_csv("p_values.csv", index=False)
