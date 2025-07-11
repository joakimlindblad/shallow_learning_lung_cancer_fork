
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier


from sklearn.svm import SVC
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.inspection import permutation_importance

import collections
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from scipy.stats import ttest_rel

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge, Lasso



def get_feature_importance(model, column_names):
    if hasattr(model, 'coef_'):
        # For Logistic Regression and similar models
        importance = model.coef_[0]  # Get the coefficients
    elif hasattr(model, 'feature_importances_'):
        # For models like Random Forest and XGBoost
        importance = model.feature_importances_
    elif hasattr(model.named_steps['regressor'], 'coef_'):
        importance = model.named_steps['regressor'].coef_[0]
        
    else:
        raise ValueError("Model does not have coefficients or feature importances.")
    
    return pd.DataFrame({'feature': column_names, 'importance': importance})


def plot_feature_importances(importances, feature_names, title='Median Feature Weight Across Folds', xlabel='Median Feature Weight'):
    sorted_indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_feature_names, sorted_importances, color='skyblue')
    plt.xlabel(xlabel)
    plt.title(title)
    plt.gca().invert_yaxis()  # Reverse the y-axis to have the most important feature on top
    plt.tight_layout()
    

class RandomClassifier:
    def fit(self, X, y):
        # Store the proportion of labels in the training set
        self.label_proportions = y.value_counts(normalize=True)

    def predict(self, X):
        # Randomly choose labels based on the stored proportions
        return np.random.choice(self.label_proportions.index, size=len(X), p=self.label_proportions.values)

    def predict_proba(self, X):
        # Return the probability distribution for each input sample
        # Each sample gets the same probability distribution based on the label proportions
        return np.tile(self.label_proportions.values, (len(X), 1))



def train_test(model, modelname, X_train, y_train, X_test, y_test, bps_test, bps_train, column_names=None):

    

    
    # Predict labels for the test set
    if modelname == "linreg":
        model.fit(X_train, y_train['time'])
        y_probas = model.predict(X_test)
        
        y_pred = y_probas >= 1717.0
    else:
        model.fit(X_train, y_train['label'])
        y_pred = model.predict(X_test)
        y_probas = np.array(model.predict_proba(X_test))[:,1]
    y_pred_test = y_pred
    y_probas_test = y_probas 
    # Calculate accuracy and AUC for Random Forest
    accuracy = accuracy_score(y_test['label'], y_pred)
    auc = roc_auc_score(y_test['label'], y_probas)
    
    bps_test["group"].append(modelname)
    bps_test["AUC_test"].append(auc)
    bps_test["Accuracy_test"].append(accuracy)


    tn, fp, fn, tp = confusion_matrix(y_test["label"], y_pred).ravel()
    # Compute PPV (Precision)
    ppv = precision_score(y_test["label"], y_pred)
    # Compute NPV
    npv = tn / (tn + fn)
    sensitivity = recall_score(y_test["label"], y_pred)  # Directly using recall_score
    # Compute Specificity (TNR)
    specificity = tn / (tn + fp)

    bps_test["Test sensitivity"].append(sensitivity)
    bps_test["Test specificity"].append(specificity)
    bps_test["Test PPV"].append(ppv)
    bps_test["Test NPV"].append(npv)

    # Predict labels for the test set
    if modelname == "linreg":
        y_probas = model.predict(X_train)
        y_pred = (y_probas >= 1717.0).astype(int)
    else:
        y_pred = model.predict(X_train)
        y_probas = np.array(model.predict_proba(X_train))[:,1]

    # Calculate accuracy and AUC for Random Forest
    accuracy = accuracy_score(y_train['label'], y_pred)
    auc = roc_auc_score(y_train['label'], y_probas)
    
    bps_train["group"].append(modelname)
    bps_train["AUC_train"].append(auc)
    bps_train["Accuracy_train"].append(accuracy)

    if modelname in ["LogReg", "XGB", "linreg"]:
        feature_importance = get_feature_importance(model.best_estimator_, column_names)
        return feature_importance, y_pred_test, y_probas_test
    else:
        return None
    #return accuracy, auc


def train_test_permutation(model, modelname, X_test, y_test, feature_importance, trainmodel=True):#, X_train, y_train, X_test, y_test, feature_importance):
    if trainmodel:
        model.fit(X_train, y_train['label'])
    
    perm_importance = permutation_importance(model, X_test, y_test["label"], n_repeats=10, random_state=42, n_jobs=-1)
    for i,c in enumerate(X_test.columns):
        feature_importance[c].append(perm_importance.importances_mean[i])
    """feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance_Mean': perm_importance.importances_mean,
        'Importance_Std': perm_importance.importances_std
    })"""
    

def get_most_important_features(model, X, y, n_splits=10):

    importance_stats = []

    feature_importance = {}
    for c in X.columns:
        feature_importance[c] = []
        
    test_size = 0.1
    stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    for train_index, test_index in stratified_split.split(X, y['label']):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()

        # Fit and transform the training data (learn mean and std from training data, then scale it)
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_test.columns)
        # Transform the test data (use the same mean and std as training data)
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
        #model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)
        train_test_permutation(model, "LogReg", X_test_scaled, y_test, feature_importance, True)
    
        #model = XGBClassifier(objective="binary:logistic",eval_metric='mlogloss')
        #train_test_permutation(model, "XGB", X_train_scaled, y_train, X_test_scaled, y_test, feature_importance)
        #xgb_accuracies.append(accuracy)
        #xgb_auc_scores.append(auc)
    

    feature_importance = pd.DataFrame(feature_importance).mean()

    return feature_importance.nlargest(10).index.tolist()


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    
    # Fit and transform the training data (learn mean and std from training data, then scale it)
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    # Transform the test data (use the same mean and std as training data)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return scaler, X_train_scaled, X_test_scaled

def init_logreg():
    log_reg_param_grid = {
        'C': np.logspace(-4, 4, 10),#[0.1, 1, 10],#(-4, 4, 10),  # Try a wide range of regularization strengths
        'penalty': ['l1', 'l2'],      # Test both L1 and L2 regularization
        'solver': ['liblinear', 'saga'],  # Solvers that support L1 and L2 penalties
        'max_iter': [10000]   # Ensure enough iterations for convergence
    }
    log_reg = LogisticRegression(random_state=42)
    model = GridSearchCV(log_reg, log_reg_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    return model

def init_linreg():
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),  # Generates polynomial and interaction features
        ('scaler', StandardScaler()),    # Standardizes features
        ('regressor', Ridge())           # Ridge Regression
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        'poly__degree': [1, 2, 3],            # Test polynomial degrees (quadratic, cubic, etc.)
        'regressor__alpha': [0.01, 0.1, 1, 10],  # Regularization strength
        'regressor__fit_intercept': [True]   # Ensure intercept is fit
    }
    
    lin_reg = LinearRegression()
    model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return model



def init_xgb():

    #model = XGBClassifier(objective="binary:logistic",eval_metric='mlogloss')
    model = XGBClassifier(
        objective="binary:logistic",eval_metric='mlogloss',
        max_depth=2,          # Control the depth of trees
        learning_rate=0.1,    # Step size for each boosting iteration
        n_estimators=100,     # Number of boosting rounds
        reg_alpha=0.2,        # L1 regularization (alpha)
        reg_lambda=1.0,       # L2 regularization (lambda)
        subsample=0.8,        # Proportion of data used per tree
        colsample_bytree=0.8,  # Proportion of features used per tree
        n_jobs=10
    )

    
    xgb = XGBClassifier()

    # Define the hyperparameter grid
    
    


    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [1, 5],
        'reg_alpha': [1, 5, 10],
        'reg_lambda': [1, 5, 10],
    }

    # Initialize GridSearchCV for XGBoost
    model = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=100,                   # Only test 20 combinations
        scoring='accuracy',
        n_jobs=-1,                    # Use all available cores
        cv=5,
        refit=True,
        random_state=42
    )
    #model = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    return model


def init_svm():
    # Define the parameter grid for SVM
    svm_param_grid = {
        'C': [0.1, 1],#np.logspace(-4, 4, 10),      # Wide range of regularization strengths
        'kernel': ['linear', 'rbf'],  # Test linear, RBF, and polynomial kernels
        'gamma': ['scale', 'auto'],       # Different kernel coefficient settings
    }

    # Initialize the SVM model
    svm = SVC(probability=True,random_state=42)

    # Set up GridSearchCV with 5-fold cross-validation
    model = GridSearchCV(svm, svm_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    return model
#    return accuracy, auc

# Load the dataset from a CSV file

df = pd.read_csv("patient_densities_morphologies.csv")
#df = pd.read_csv("patient_cluster_statistics.csv")

# Define the features and survival data
X = df.drop(columns=["ID", 'time', 'event', 'label'])  # Features


#X = X.drop(columns=['CD4_Single Nucleus Axis Ratio', 'CD4_Single Nucleus Compactness', 'CD4_Single Nucleus Area', 'CD4_Treg Nucleus Axis Ratio', 'CD4_Treg Nucleus Compactness', 'CD4_Treg Nucleus Area', 'CD8_Single Nucleus Axis Ratio', 'CD8_Single Nucleus Compactness', 'CD8_Single Nucleus Area', 'CD8_Treg Nucleus Axis Ratio', 'CD8_Treg Nucleus Compactness', 'CD8_Treg Nucleus Area', 'B_cells Nucleus Axis Ratio', 'B_cells Nucleus Compactness', 'B_cells Nucleus Area', 'Stroma_other Nucleus Axis Ratio', 'Stroma_other Nucleus Compactness', 'Stroma_other Nucleus Area'])
column_names = X.columns.values.tolist()
y = df[["ID", 'time', 'event', 'label']]  # Survival time, event status, and binary label

# Define the StratifiedShuffleSplit cross-validator
n_splits = 100
test_size = 0.1

stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
split_folder = "/data2/love/multiplex_cancer_cohorts/patient_and_samples_data/lung_cancer_BOMI2_dataset/binary_survival_prediction/100foldcrossvalrepeat/"
#split_folder = "/data2/love/multiplex_cancer_cohorts/patient_and_samples_data//backup_lung_cancer_dataset/binary_survival_prediction/100foldcrossvalrepeat/"
#split_folder = "/data2/love/multiplex_cancer_cohorts/patient_and_samples_data/lung_cancer_BOMI2_dataset/binary_survival_prediction/10foldcrossval/"
split_files = os.listdir(split_folder)
num_splits = len([x for x in split_files if "test" in x])




#clinical_parameters = [column_names[i] for i in [1, 4, 5]]

#morphologies = [x for x in morphologies if ("Tumor" in x) or ("Stroma" in x)]


#densities = ["Tumor CD4_Single", "Stroma B_cells", "Stroma CD8_Single", "Stroma CD8_Treg", "Stroma CD4_Treg"]
#densities = [x for x in densities if ("Tumor" in x) or ("Stroma" in x)]
clinical_parameters = column_names[:6]
morphologies = column_names[6:(6+3)]
densities = column_names[(6+3):]
part_names = {"clinical parameters": clinical_parameters, "morphologies": morphologies, "densities": densities}

name_combination_list = [["clinical parameters","morphologies","densities"],
                ["clinical parameters","morphologies"],
                ["clinical parameters","densities"],
                ["morphologies","densities"],
                ["clinical parameters"],
                ["morphologies"],
                ["densities"]
                         ]
"""




clinical_parameters = column_names[:7]
densities = column_names[7:7*3]
morphologies = column_names[7*3:]

part_names = {"clinical parameters": clinical_parameters, "morphologies": morphologies, "densities": densities}

name_combination_list = [["clinical parameters","morphologies","densities"],
                ["clinical parameters","morphologies"],
                ["clinical parameters","densities"],
                ["morphologies","densities"],
                ["clinical parameters"],
                ["morphologies"],
                ["densities"]]

"""


"""
part_names = {"clinical parameters": clinical_parameters, "densities": densities}
name_combination_list = [["clinical parameters","densities"],
                ["clinical parameters"],
                ["densities"]]
"""


#X[densities] = (X[densities] > X[densities].median()).astype(int)


experiment_names = ["_".join(name_comb) for name_comb in name_combination_list]
combinations = [part_names[part_name] for name_combination in name_combination_list for part_name in name_combination]


results_dict = {}
results_dict_titles = ["Experiment", "Model", "Train accuracy mean", "Train accuracy std", "Train AUC mean", "Train AUC std",
                       "Test accuracy mean", "Test accuracy std", "Test accuracy sem", "Test AUC mean", "Test AUC std", "Test AUC sem", "Test sensitivity", "Test specificity", "Test NPV", "Test PPV"]

for t in results_dict_titles:
    results_dict[t] = []


experiment_results_acc = {}
experiment_results_auc = {}



for name_comb in name_combination_list:
    experiment_name = "_".join(name_comb)
    column_names = [part_names[part] for part in name_comb]
    column_names =  sum(column_names, [])
    
    predictions = {}
    logits = {}
    for ID in df["ID"]:
        predictions[ID] = []
        logits[ID] = []
    preds_logits = []
    print(experiment_name)
    print(column_names)
    
    #for experiment_name, column_names in zip(experiment_names, combinations):
    
    bps_test = {"group": [], "AUC_test": [], "Accuracy_test": [], "Test sensitivity":[], "Test specificity":[], "Test NPV":[], "Test PPV":[]}
    bps_train = {"group": [], "AUC_train": [], "Accuracy_train": []}

    feature_importance_dfs_logreg = []
    feature_importance_dfs_xgb = []
    X_experiment = X[column_names]
    permutation_importance_data = {}
    for c in X_experiment.columns:
        permutation_importance_data[c] = []

    for split in tqdm(range(num_splits)):
        train_ids = pd.read_csv(os.path.join(split_folder, "split_" + str(split) + "_train_val.csv"))["ID"]
        test_ids = pd.read_csv(os.path.join(split_folder, "split_" + str(split) + "_test.csv"))["ID"]
        
        train_index = df["ID"].isin(train_ids)
        test_index = df["ID"].isin(test_ids)
        
        X_train, X_test = X_experiment[train_index], X_experiment[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        
        ###### log reg #####
        #topfeats = get_most_important_features(X_train_scaled, y_train, n_splits=10)
        
        model = init_logreg()
        #model = init_linreg()
        #feature_importance_dfs_logreg.append(train_test(model, "LogReg", X_train_scaled[topfeats], y_train,
        #                                                X_test_scaled[topfeats], y_test, bps_test, bps_train, topfeats))
        feature_weights, preds, probas = train_test(model, "LogReg", X_train_scaled, y_train, X_test_scaled, y_test,
                                                    bps_test, bps_train, column_names)
        feature_importance_dfs_logreg.append(feature_weights)
        
        
        """
        ####  XGB #####
        model = init_xgb()
        feature_weights, preds, probas = train_test(model, "LogReg", X_train_scaled, y_train, X_test_scaled, y_test,
                                                    bps_test, bps_train, column_names)
        feature_importance_dfs_logreg.append(feature_weights)

        """
        
        IDs = df[test_index]["ID"].values
        
        split_results = pd.DataFrame({
            "ID": test_ids,
            "split": [split]*len(test_ids),
            "prediction": preds,
            "logit": probas
        })
        preds_logits.append(split_results)
        
        for ID, pred, proba in zip(IDs, preds, probas):
            predictions[ID].append(pred)
            logits[ID].append(proba)
        
        #train_test_permutation(model, "logreg", X_test_scaled[top5feats], y_test, permutation_importance_data)
        #train_test_permutation(model, "logreg", X_test_scaled, y_test, permutation_importance_data)
        
        
        
        
        
        
        """
        ##### random classifier #####
        model = RandomClassifier()
        train_test(model, "Random", X_train_scaled, y_train, X_test_scaled, y_test, bps_test, bps_train)
        
        """
        ##########  SVM   #############

        #model = init_svm()
        #feature_importance_dfs_logreg.append(train_test(model, "SVM", X_train_scaled, y_train, X_test_scaled, y_test, bps_test, bps_train,column_names))

        """
        ########## KNN  ############

        knn = KNeighborsClassifier()
        knn_param_grid = {
            'n_neighbors': [5, 10, 20, 30, 40],             # More neighbors will smooth the model and reduce overfitting
        }
        model = GridSearchCV(knn, knn_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        #model = KNeighborsClassifier(n_neighbors=10)
        train_test(model, "KNN", X_train_scaled, y_train, X_test_scaled, y_test, bps_test, bps_train)
        """
        
        
        
    bps_test = pd.DataFrame(bps_test)
    bps_train = pd.DataFrame(bps_train)

    modelname = "LogReg"#"linreg"#
    
    train_stats = bps_train[bps_train["group"] == modelname]
    test_stats = bps_test[bps_test["group"] == modelname]
    
    results_dict["Experiment"].append(experiment_name)
    results_dict["Model"].append(modelname)
    
    results_dict["Train accuracy mean"].append(train_stats["Accuracy_train"].mean())
    results_dict["Train accuracy std"].append(train_stats["Accuracy_train"].std())
    results_dict["Train AUC mean"].append(train_stats["AUC_train"].mean())
    results_dict["Train AUC std"].append(train_stats["AUC_train"].std())
    
    
    results_dict["Test accuracy mean"].append(test_stats["Accuracy_test"].mean())
    results_dict["Test accuracy std"].append(test_stats["Accuracy_test"].std())
    results_dict["Test accuracy sem"].append(test_stats["Accuracy_test"].std() / np.sqrt(len(test_stats["Accuracy_test"])))
    results_dict["Test AUC mean"].append(test_stats["AUC_test"].mean())
    results_dict["Test AUC std"].append(test_stats["AUC_test"].std())
    results_dict["Test AUC sem"].append(test_stats["AUC_test"].std() / np.sqrt(len(test_stats["AUC_test"])))
    results_dict["Test sensitivity"].append(test_stats["Test sensitivity"].mean())
    results_dict["Test specificity"].append(test_stats["Test specificity"].mean())
    results_dict["Test PPV"].append(test_stats["Test PPV"].mean())
    results_dict["Test NPV"].append(test_stats["Test NPV"].mean())
    
    print("average test accuracy:", test_stats["Accuracy_test"].mean())
    print("average test AUC:", test_stats["AUC_test"].mean())
    # Use collections.Counter to count the occurrences of each string

    experiment_results_acc[experiment_name] = test_stats["Accuracy_test"].values
    experiment_results_auc[experiment_name] = test_stats["AUC_test"].values

    min_val = min(bps_train['AUC_train'].min(), bps_train['Accuracy_train'].min(), 
              bps_test['AUC_test'].min(), bps_test['Accuracy_test'].min())
    max_val = max(bps_train['AUC_train'].max(), bps_train['Accuracy_train'].max(), 
              bps_test['AUC_test'].max(), bps_test['Accuracy_test'].max())
    min_val -= 0.1
    max_val += 0.1
    #plt.show()

    """plt.figure()
    sns.boxplot(x='group', y='AUC_train', data=bps_train)
    plt.title('AUC_train')
    plt.ylim(min_val, max_val)

    plt.figure()
    sns.boxplot(x='group', y='Accuracy_train', data=bps_train)
    plt.title('Accuracy_train')
    plt.ylim(min_val, max_val)

    plt.figure()
    sns.boxplot(x='group', y='AUC_test', data=bps_test)
    plt.title('AUC_test')
    plt.ylim(min_val, max_val)

    plt.figure()
    sns.boxplot(x='group', y='Accuracy_test', data=bps_test)
    plt.title('Accuracy_test')
    plt.ylim(min_val, max_val)"""
    #print(feature_importance_dfs_logreg)
    
    all_importances = pd.concat(feature_importance_dfs_logreg)
    aggregation_name = "mean"
    # Step 2: Compute the average feature importance
    average_importance = all_importances.groupby('feature')['importance'].mean().reset_index()
    # Step 3: Sort the features by average importance
    average_importance = average_importance.sort_values(by='importance', ascending=False)
    plot_feature_importances(average_importance['importance'].values, average_importance['feature'].values,
                             title=aggregation_name + ' Feature Weight Across Folds',
                             xlabel=aggregation_name + ' Feature Weight')
    
    plt.savefig("plots/" + aggregation_name + "_feature_weight"+ experiment_name + ".png")
    

    """
    permutation_importance_data = pd.DataFrame(permutation_importance_data).mean().values
    plot_feature_importances(permutation_importance_data, X_experiment.columns.values,
                             title=aggregation_name + ' Permutation Feature Importance Across Folds',
                             xlabel=aggregation_name + ' Permutation Importance')
    plt.savefig("plots/median_feature_permutatio_importance" + experiment_name + ".png")
    """

    predictions = pd.DataFrame(predictions)
    logits = pd.DataFrame(logits)
    preds_logits = pd.concat(preds_logits, ignore_index=False)
    preds_logits = preds_logits.groupby("ID").agg({
        "prediction": "mean",  # Average prediction over splits
        "logit": "mean"        # Average logit over splits
    }).rename(columns={"prediction": "mean_prediction", "logit": "mean_logit"})
    preds_logits.to_csv(os.path.join("preds", experiment_name + "_preds_logits.csv"), index=False)
    predictions.to_csv(os.path.join("preds", experiment_name + "_predictions.csv"), index=False)
    logits.to_csv(os.path.join("preds", experiment_name + "_logits.csv"), index=False)
    
    
"""
all_importances = pd.concat(feature_importance_dfs_xgb)
# Step 2: Compute the average feature importance
average_importance = all_importances.groupby('feature')['importance'].mean().reset_index()
# Step 3: Sort the features by average importance
average_importance = average_importance.sort_values(by='importance', ascending=False)
plot_feature_importances(average_importance['importance'].values, average_importance['feature'].values)
"""

results = pd.DataFrame(results_dict)
results.to_csv("results_shallow_learning.csv", index=False)
plt.show()

#get_most_important_features(X, y)
#print(feature_importance.columns)
#print(feature_importance.mean().values)
#print(feature_importance.mean()*1000)


p_value_stats = {"experiment": [], "Accuracy": [], "AUC": []}

experiment_results_acc = pd.DataFrame(experiment_results_acc)
experiment_results_auc = pd.DataFrame(experiment_results_auc)

clinical_results_acc = experiment_results_acc["clinical parameters"]
clinical_results_auc = experiment_results_auc["clinical parameters"]

for name_comb in name_combination_list:
    experiment_name = "_".join(name_comb)
    results_acc = experiment_results_acc[experiment_name]
    results_auc = experiment_results_auc[experiment_name]
    t_stat, p_value_acc = ttest_rel(results_acc, clinical_results_acc)
    t_stat, p_value_auc = ttest_rel(results_auc, clinical_results_auc)
    
    p_value_stats["experiment"].append(experiment_name)
    p_value_stats["Accuracy"].append(p_value_acc)
    p_value_stats["AUC"].append(p_value_auc)
    
pd.DataFrame(p_value_stats).to_csv("p_values.csv", index=False)
