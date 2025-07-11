import csv
import pandas as pd
#from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#import kaplanmeier as km
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.stats import entropy as shannon_entropy

#import kaplanmeier as km
from lifelines import CoxPHFitter
#from survive import SurvivalData, KaplanMeier
import forestplot as fp

from lifelines.statistics import logrank_test

import seaborn as sns
from lifelines import KaplanMeierFitter

import os

from functools import reduce

from sklearn.metrics import accuracy_score, roc_auc_score


"""
def get_best_cutoff(data, feature):
    df = data.copy()
    best_p_value = 10
    #data dataframe(features, time, event)
    for cutoff in np.percentile(df[feature], np.arange(10, 90, 1)):  # Try cutoffs from 10th to 90th percentile
        df['group'] = (df[feature] > cutoff).astype(int)  # Split into two groups: 0 (low), 1 (high)
    
        # Perform log-rank test
        results = logrank_test(df['time'][df['group'] == 0], 
                               df['time'][df['group'] == 1],
                               event_observed_A=df['event'][df['group'] == 0],
                               event_observed_B=df['event'][df['group'] == 1])
    
        if results.p_value < best_p_value:
            best_p_value = results.p_value
            best_cutoff = cutoff

    return best_cutoff
"""


def dichotomize_clinpars(data):
    data["Age"] = data["Age"].map(lambda x: 1 if x > 70 else 0)
    data["Performance status"] = data["Performance status"].map(lambda x: 1 if x > 0 else 0)
    data["Stage"] = data["Stage"].map(lambda x: 1 if x > 1 else 0)
    return data

def cencor_after5years(data):
    data.loc[data["time"] > 5*365, "event"] = 0
    return data

def get_best_cutoff(data, feature):
    df = data.copy()
    best_p_value = 10
    best_cutoff = None

    # Iterate over percentiles from 10th to 90th
    for cutoff in np.percentile(df[feature], np.arange(10, 90, 1)):#100/298)):
        # Split into two groups based on the cutoff
        df['group'] = (df[feature] > cutoff).astype(int)
        
        # Check if both groups have variance (at least one entry in each group)
        if df['group'].nunique() < 2:
            continue  # Skip this cutoff if it results in a single group
        
        # Perform log-rank test
        results = logrank_test(df['time'][df['group'] == 0], 
                               df['time'][df['group'] == 1],
                               event_observed_A=df['event'][df['group'] == 0],
                               event_observed_B=df['event'][df['group'] == 1])
        
        # Update the best cutoff if p-value improves
        if results.p_value < best_p_value:
            best_p_value = results.p_value
            best_cutoff = cutoff

    # Return the best cutoff, or None if no valid cutoff was found
    return best_cutoff

def get_all_best_cutoffs(data, experiments):
    #for exp in experiments:
    #    data = data.drop(columns=[exp + " pred"])
    data = data.drop(columns=["ID", "label", 'LUAD', 'Age', 'Gender', 'Smoking', 'Stage', 'Performance status'])
    cutoffs = {}
    for column in data.columns:
        if column in ["time", "event"]:
            continue
        cutoff = get_best_cutoff(data, column)
        cutoffs[column] = cutoff

    return cutoffs



def multivariate_coxregression(data, p_values):
    data = data.drop(columns=["ID", "label"])# + [x for x in data.columns if x.endswith("logit")])
    columns = [x for x in p_values if ((p_values[x] < 0.05) and (not x.endswith("pred")) and (not x.endswith("logit")) )]
    data = data[["time", "event"] + columns]
    cph = CoxPHFitter()
    cph.fit(data, duration_col="time", event_col="event")
    summary = cph.summary.reset_index()
    print(summary)
    summary = summary[["covariate", 'coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', "p"]]
    summary.columns = ['Feature', 'Coefficient', 'Hazard Ratio', 'Lower CI', 'Upper CI', "p"]

    print(summary)
    fig = plt.figure(figsize=(12, 8))  # Adjust figure size
    fig = fp.forestplot(summary, estimate="Hazard Ratio", ll="Lower CI", hl="Upper CI", pval="p", ylabel="confidence interval", xlabel="Coefficient", varlabel="Feature",flush=False,figsize=(12,8), **{'fontfamily': 'sans-serif'} )
    plt.axvline(x=1, color='red', linestyle='--', label='Vertical Line')

    xmax = np.amax(summary["Upper CI"].values)
    xmin = np.amin(summary["Lower CI"].values)
    xlim = np.amax([np.abs((xmin-1)), np.abs((xmax-1))])

    plt.tight_layout()
    
    plt.savefig("multivariate_cox.png")

def univariate_coxregression(data, filename="univariatecox_features.png"):
    """
    Perform univariate Cox regression for each feature independently.

    Parameters:
    - data: pandas DataFrame containing 'time', 'event', and features.

    Returns:
    - summary: pandas DataFrame containing the coefficients, hazard ratios, and p-values for each feature.
    """

    # Drop non-relevant columns (you may adjust the columns to drop)
    data = data.drop(columns=["ID", "label"], errors='ignore')  # Adjust as needed
    #densities = data.columns.values[(2+6+3):]
    #data[densities] = (data[densities] > data[densities].median()).astype(int)
    # Initialize a CoxPHFitter object
    cph = CoxPHFitter()

    # Initialize a list to collect results for each feature
    results = []
    p_values = {}
    # Loop over each feature in the DataFrame (except 'time' and 'event')
    for feature in data.columns:
        if feature not in ['time', 'event']:  # Skip time and event columns
            print(feature)
            # Select only the 'time', 'event', and current feature
            temp_data = data[['time', 'event', feature]]
            if temp_data[feature].var() == 0:
                print(f"Skipping feature {feature}: no variance")
                print(temp_data[feature].unique())
                continue
            # Fit the Cox model for the current feature
            cph.fit(temp_data, duration_col='time', event_col='event')
            
            # Get the summary for the current feature
            summary = cph.summary.loc[feature, ['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
            
            # Append the results to the list
            results.append([feature, summary['coef'], summary['exp(coef)'], summary['exp(coef) lower 95%'], summary['exp(coef) upper 95%'], summary['p']])
            p_values[feature] = summary["p"]
    # Create a summary DataFrame for all features
    summary_df = pd.DataFrame(results, columns=['Feature', 'Coefficient', 'Hazard Ratio', 'Lower CI', 'Upper CI', 'p-value'])

    # Print the summary
    print(summary_df)
    summary_features = summary_df[~summary_df["Feature"].str.endswith(("pred", "logit"), na=False)]
    summary_models = summary_df[summary_df["Feature"].str.endswith("pred", na=False)]


    
    # Plot the forest plot
    fig = plt.figure(figsize=(12, 8))
    fig = fp.forestplot(
        summary_features,
        estimate="Hazard Ratio",
        ll="Lower CI",
        hl="Upper CI",
        pval="p-value",
        ylabel="Confidence Interval",
        xlabel="Hazard Ratio",
        varlabel="Feature",
        flush=False,
        figsize=(12, 8),
        **{'fontfamily': 'sans-serif', 'fontsize': 10}
    )
    plt.axvline(x=1, color='red', linestyle='--', label='Vertical Line')
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig("univariate_cox_features.png", bbox_inches='tight')

        # Plot the forest plot
    fig = plt.figure(figsize=(12, 8))
    fig = fp.forestplot(
        summary_models,
        estimate="Hazard Ratio",
        ll="Lower CI",
        hl="Upper CI",
        pval="p-value",
        ylabel="Confidence Interval",
        xlabel="Hazard Ratio",
        varlabel="Feature",
        flush=False,
        figsize=(12, 8),
        **{'fontfamily': 'sans-serif', 'fontsize': 10}
    )
    plt.axvline(x=1, color='red', linestyle='--', label='Vertical Line')
    # Adjust layout and save the plot
    plt.tight_layout()
    #plt.xlim(0.25,1.75)
    plt.savefig("univariate_cox_models.png", bbox_inches='tight')
    
    
    return p_values



def plot_kaplan_meier(data, time_col, event_col, feature_col, name):
    """
    Plot Kaplan-Meier survival curves for a binary feature.
    
    Parameters:
    - data: pandas DataFrame containing the data.
    - time_col: str, column name for survival times.
    - event_col: str, column name for event occurrence (1 = event, 0 = censored).
    - feature_col: str, column name of the binary feature (0 or 1).
    
    Returns:
    - None
    """
    kmf = KaplanMeierFitter()
    print(feature_col)
    plt.figure(figsize=(10, 6))
    
    for value in [0, 1]:
        subset = data[data[feature_col] == value]
        print(len(subset))
        kmf.fit(subset[time_col], event_observed=subset[event_col], label=f'{feature_col} = {value}')
        kmf.plot_survival_function(ci_show=True)


    fraction_positive = data[feature_col].mean()
    fraction_text = f"Fraction of positive predictions: {fraction_positive:.2f}"
     
    # Add the text to the plot
    plt.text(0.7, 0.1, fraction_text, transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.5))

    
    
    plt.title('Kaplan-Meier Curve')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("km_plots", name))



    
    # Perform log-rank test
    group0 = data[data[feature_col] == 0]
    group1 = data[data[feature_col] == 1]
    results = logrank_test(group0[time_col], group1[time_col], 
                           event_observed_A=group0[event_col], event_observed_B=group1[event_col])
    
    print(f"Log-rank test p-value: {results.p_value}")



def get_patient_preds():
    names = [x.replace("_LogReg_predictions.csv", "") for x in os.listdir("./preds/") if x .endswith("_LogReg_predictions.csv")]
    all_model_preds = []
    for experiment_name in names:
        preds = pd.read_csv(os.path.join("./preds/", experiment_name + "_LogReg_predictions.csv"))#.iloc[:,1:]
        
        logits = pd.read_csv(os.path.join("./preds/", experiment_name + "_LogReg_logits.csv"))#.iloc[:,1:]
        #print(preds)
        
        preds = preds.mean(axis=0)#preds.median(axis=0).astype(int)#
        
        #print(preds)
        preds = preds.reset_index()
        
        preds.columns = ["ID", experiment_name + " pred"]
        
        logits = logits.mean(axis=0)
        logits = logits.reset_index()

        logits.columns = ["ID", experiment_name + " logit"]
        patient_preds = pd.merge(logits, preds, on="ID")
        
        all_model_preds.append(patient_preds)

    all_model_preds = reduce(lambda left, right: pd.merge(left, right, on='ID'), all_model_preds)
        
    return names, all_model_preds


def get_extended_preds():
    names = [x.replace("_predictions.csv", "") for x in os.listdir("./preds/") if x .endswith("_predictions.csv")]
    all_model_preds = []
    for experiment_name in names:
        print(experiment_name)
        preds = pd.read_csv(os.path.join("./preds/", experiment_name + "_predictions.csv"))
        logits = pd.read_csv(os.path.join("./preds/", experiment_name + "_logits.csv"))
        preds = preds.melt(var_name="ID", value_name="pred")
        logits = logits.melt(var_name="ID", value_name="logit")
        preds.columns = ["ID", experiment_name + " pred"]
        logits.columns = ["ID", experiment_name + " logit"]
        patient_preds = pd.merge(logits, preds.drop(columns=["ID"]), left_index=True, right_index=True)#, on="ID")
        all_model_preds.append(patient_preds)
        print(patient_preds)
    print("merging")
    
    model_pred1 = all_model_preds[0]
    all_model_preds = all_model_preds[1:]
    for model_pred in all_model_preds:
        model_pred1 = pd.merge(model_pred1, model_pred.drop(columns=["ID"]), left_index=True, right_index=True)

    all_model_preds = model_pred1
    #all_model_preds = reduce(lambda left, right: pd.merge(left, right, on='ID', how="outer"), all_model_preds)
    print("merge done")
    return names, all_model_preds


def get_patient_pred():
    names = [x.replace("_predictions.csv", "") for x in os.listdir("./preds/") if x .endswith("_predictions.csv")]
    all_model_preds = []
    for experiment_name in names:
        pred_logits = pd.read_csv(os.path.join("./preds/", experiment_name + "preds_logits.csv"))
        all_model_preds.append(preds_logits)
                                  

experiment_names, all_models_preds = get_patient_preds()#get_extended_preds()##

        
patient_data = pd.read_csv("patient_densities_morphologies.csv")
patient_data['ID'] = patient_data['ID'].astype(float)
print(all_models_preds)
all_models_preds['ID'] = all_models_preds['ID'].astype(float)

patient_data = dichotomize_clinpars(patient_data)
patient_data = cencor_after5years(patient_data)

patient_data_model_preds = pd.merge(patient_data, all_models_preds, on="ID")
print(patient_data_model_preds)
print(patient_data_model_preds[["ID", "label", "clinical parameters logit", "clinical parameters pred"]])

patient_data = patient_data_model_preds


#print(patient_data["densities pred"].values)
cutoffs = get_all_best_cutoffs(patient_data, experiment_names)
for feature in cutoffs:
    cutoff = cutoffs[feature]
    patient_data[feature] = (patient_data[feature] > cutoff).astype(int)
#print(patient_data["densities pred"].values)


#print("accuracy:", accuracy_score(patient_data_model_preds["label"], patient_data_model_preds["morphologies pred"]))
print("accuracy:", roc_auc_score(patient_data_model_preds["label"], patient_data_model_preds["clinical parameters pred"]))
print(patient_data_model_preds.columns.values)
p_values = univariate_coxregression(patient_data)
multivariate_coxregression(patient_data, p_values)




for feature in cutoffs:
    plot_kaplan_meier(patient_data, "time", "event", feature, feature + ".png")

for feature in ['LUAD', 'Age', 'Gender', 'Smoking', 'Stage', 'Performance status']:
    plot_kaplan_meier(patient_data, "time", "event", feature, feature + ".png")

#for name in experiment_names:
#    plot_kaplan_meier(patient_data, "time", "event", name + " pred", name + ".png")

