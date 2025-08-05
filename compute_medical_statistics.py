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
import re

from functools import reduce

from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.font_manager as fm
from forestplot_custom import simple_forestplot

def apply_feature_font(fig, feature_list, fontfamily, fontsize, fontstyle='normal', fontweight='normal'):

    font_prop = fm.FontProperties(
        family=fontfamily,
        size=fontsize,
        style=fontstyle,
        weight=fontweight
    )
    for text in fig.texts:
        text.set_fontproperties(font_prop)
def make_forestplot(
    df,
    estimate="Hazard Ratio",
    ll="Lower CI",
    hl="Upper CI",
    varlabel="Feature",
    pval="p-value",
    xlabel="Hazard ratio",
    ylabel="Confidence interval",
    fontsize=18,
    pval_fontsize=18,
    tick_fontsize=14,
    fontfamily="Arial",
    axvline=1,
    axvline_color='red',
    axvline_style='--',
    labelpad=-525,
    tight_layout=True,
):
    # Create forestplot
    fig = fp.forestplot(
        df,
        estimate=estimate,
        ll=ll,
        hl=hl,
        varlabel=varlabel,
        flush=False,
        pval=pval,
        ylabel=ylabel,
        xlabel=xlabel,
        figsize=(12,8),
        fontsize=fontsize,
        
        
    )
    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontfamily(fontfamily)
        label.set_horizontalalignment('left')
    fig.set_ylabel(ylabel, labelpad=labelpad, ha="left", fontsize=20)

    # Set font size for p-values
    for text in fig.texts:
        if re.search(r"\d", text.get_text()):
            text.set_fontsize(pval_fontsize)
    # Set x-axis tick label font size
    fig.tick_params(axis='x', labelsize=tick_fontsize)

    # Reference line at HR = 1
    plt.axvline(x=axvline, color=axvline_color, linestyle=axvline_style, label=f'HR = {axvline}')
    if tight_layout:
        plt.tight_layout()


    #apply_feature_font(fig, df[varlabel].tolist(), 'Arial', fontsize, 'italic', 'normal')
    for label in fig.get_yticklabels():
        print(label.get_fontfamily())
        #label.set_fontfamily('Arial')
        #label.set_fontstyle("italic")
        #label.set_ha("left") 

    return fig

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
    print(feature, best_p_value)
    return best_cutoff

def get_all_best_cutoffs(data, experiments):
    #for exp in experiments:
    #    data = data.drop(columns=[exp + " pred"])
    data = data.drop(columns=["ID", "label", 'LUAD', 'Age', 'Sex (Male)', 'Smoking', 'Stage', 'Performance status'])
    data = data.drop(columns=[col for col in data.columns if col.endswith("pred.")])
    data = data.drop(columns=[col for col in data.columns if col.endswith("logit")])
    cutoffs = {}
    for column in tqdm(data.columns):
        if column in ["time", "event"]:
            continue
        cutoff = get_best_cutoff(data, column)
        cutoffs[column] = cutoff

    return cutoffs

def forward_stepwise_coxregression(
    data,
    p_values,
    threshold=0.05,
    plot_path="forward_stepwise_cox_forestplot.svg"
):
    """
    Forward stepwise Cox regression with features significant in univariate Cox (p<0.05),
    except those ending with 'pred' or 'logit'.
    Starts with the single best feature, then adds features one by one if they are significant.
    Saves a forest plot and returns the model, selected features, and summary.
    """
    from lifelines import CoxPHFitter
    import matplotlib.pyplot as plt
    import forestplot as fp

    # Use only univariately significant features, except model outputs
    eligible_features = [
        f for f in p_values
        if (p_values[f] < 0.05)
        and (not f.endswith("pred."))
        and (not f.endswith("logit"))
        and (f not in ["ID", "label", "time", "event"])
        and (f in data.columns)
    ]
    if len(eligible_features) == 0:
        raise ValueError("No eligible features to include in forward stepwise Cox regression.")

    selected_features = []
    remaining_features = eligible_features.copy()
    cph = CoxPHFitter()
    best_p = 1

    # Step 1: Find the single most significant feature (lowest p in univariate)
    first_feature = min(remaining_features, key=lambda f: p_values[f])
    selected_features.append(first_feature)
    remaining_features.remove(first_feature)

    # Step 2: Add features one by one if they improve the model
    while remaining_features:
        best_feature = None
        best_feature_p = 1
        # Try adding each remaining feature and check if it is significant in multivariate
        for feat in remaining_features:
            test_features = selected_features + [feat]
            try:
                cph.fit(data[["time", "event"] + test_features], duration_col="time", event_col="event")
                pval = cph.summary.loc[feat, "p"]
                if pval < best_feature_p:
                    best_feature_p = pval
                    best_feature = feat
            except Exception:
                continue
        if best_feature is not None and best_feature_p < threshold:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break  # No more features improve the model significantly

    # Final fit and plot
    cph.fit(data[["time", "event"] + selected_features], duration_col="time", event_col="event")
    summary = cph.summary.reset_index()
    summary = summary[["covariate", 'coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', "p"]]
    summary.columns = ['Feature', 'Coefficient', 'Hazard Ratio', 'Lower CI', 'Upper CI', "p"]

    fig = plt.figure(figsize=(12, 8))
    fig = fp.forestplot(
        summary,
        estimate="Hazard Ratio",
        ll="Lower CI",
        hl="Upper CI",
        pval="p",
        ylabel="Confidence Interval",
        xlabel="Hazard Ratio",
        varlabel="Feature",
        flush=False,
        figsize=(12, 8),
        **{'fontfamily': 'sans-serif', 'fontsize': 10}
    )
    plt.axvline(x=1, color='red', linestyle='--', label='HR = 1')
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return cph, selected_features, summary


def stepwise_coxregression(
    data,
    p_values,
    threshold=0.05,
    min_features=1,
    plot_path="stepwise_cox_forestplot.png"
):
    """
    Backward stepwise Cox regression with features significant in univariate Cox (p<0.05),
    except those ending with 'pred' or 'logit'.
    Saves a forest plot and returns the model, selected features, and summary.
    """
    from lifelines import CoxPHFitter
    import matplotlib.pyplot as plt
    import forestplot as fp

    # Use only univariately significant features, except model outputs
    selected_features = [
        f for f in p_values
        if (p_values[f] < 0.05)
        and (not f.endswith("pred."))
        and (not f.endswith("logit"))
        and (f not in ["ID", "label", "time", "event"])
        and (f in data.columns)
    ]
    features = selected_features.copy()
    if len(features) == 0:
        raise ValueError("No features to include in multivariate stepwise Cox regression.")

    cph = CoxPHFitter()
    data_ = data[["time", "event"] + features].copy()

    # Backward stepwise elimination
    while len(features) > min_features:
        cph.fit(data_[["time", "event"] + features], duration_col="time", event_col="event")
        summary = cph.summary
        # Remove feature with highest p-value if above threshold
        pvals = summary["p"]
        worst_p = pvals.max()
        if worst_p < threshold:
            break
        worst_feature = pvals.idxmax()
        features.remove(worst_feature)

    # Final fit and plot
    cph.fit(data_[["time", "event"] + features], duration_col="time", event_col="event")
    summary = cph.summary.reset_index()
    summary = summary[["covariate", 'coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', "p"]]
    summary.columns = ['Feature', 'Coefficient', 'Hazard Ratio', 'Lower CI', 'Upper CI', "p"]

    fig = plt.figure(figsize=(12, 8))
    fig = fp.forestplot(
        summary,
        estimate="Hazard Ratio",
        ll="Lower CI",
        hl="Upper CI",
        pval="p",
        ylabel="Confidence Interval",
        xlabel="Hazard Ratio",
        varlabel="Feature",
        flush=False,
        figsize=(12, 8),
        **{'fontfamily': 'sans-serif', 'fontsize': 10}
    )
    plt.axvline(x=1, color='red', linestyle='--', label='HR = 1')
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return cph, features, summary


def multivariate_coxregression(data, p_values):
    data = data.drop(columns=["ID", "label"])# + [x for x in data.columns if x.endswith("logit")])
    columns = [x for x in p_values if ((p_values[x] < 0.05) and (not x.endswith("pred.")) and (not x.endswith("logit")) )]
    data = data[["time", "event"] + columns]
    cph = CoxPHFitter()
    cph.fit(data, duration_col="time", event_col="event")
    summary = cph.summary.reset_index()
    #print(summary)
    summary = summary[["covariate", 'coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', "p"]]
    summary.columns = ['Feature', 'Coefficient', 'Hazard Ratio', 'Lower CI', 'Upper CI', "p"]


    """fig = fp.forestplot(summary,  # the dataframe with results data
                        estimate="Hazard Ratio",
                        ll="Lower CI",
                        hl="Upper CI",
                        varlabel="Feature",
                        flush=False,
                        pval="p",
                        ylabel="Confidence interval",  # y-label title
                        xlabel="Hazard ratio",  # x-label title
                        figsize=(12,8),
                        fontsize=18,
                        )
    fig.set_ylabel("Confidence Interval", labelpad=-210, ha="left")
    plt.axvline(x=1, color='red', linestyle='--', label='HR = 1')
    plt.tight_layout()"""
    """fig = make_forestplot(summary, estimate="Hazard Ratio",
                        ll="Lower CI",
                        hl="Upper CI",
                        varlabel="Feature",
                        pval="p",
                        ylabel="Confidence interval",  # y-label title
                        xlabel="Hazard ratio",  # x-label title
                        labelpad=-155
                        )"""
    fig = simple_forestplot(summary, est_col="Hazard Ratio",
                            ll_col="Lower CI",
                            hl_col="Upper CI",
                            feature_col="Feature",
                            pval_col="p",
                            xlabel="Hazard ratio",  # x-label title
                            )
    
    plt.savefig("multivariate_cox.svg", bbox_inches="tight",)

def univariate_coxregression(data, filename="univariatecox_features.svg"):
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
            #print(feature)
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
    #print(summary_df)
    summary_features = summary_df[~summary_df["Feature"].str.endswith(("pred.", "logit"), na=False)]
    summary_models = summary_df[summary_df["Feature"].str.endswith("pred.", na=False)]

    # Plot the forest plot

    fig = simple_forestplot(summary_features, est_col="Hazard Ratio",
                        ll_col="Lower CI",
                        hl_col="Upper CI",
                        feature_col="Feature",
                        pval_col="p-value",
                        xlabel="Hazard ratio",  # x-label title
                            )
    
    """fig = make_forestplot(summary_features, estimate="Hazard Ratio",
                        ll="Lower CI",
                        hl="Upper CI",
                        varlabel="Feature",
                        pval="p-value",
                        ylabel="Confidence interval",  # y-label title
                        xlabel="Hazard ratio",  # x-label title
                        labelpad=-170
                        )"""
    
    plt.savefig("univariate_cox_features.svg", bbox_inches='tight')

        # Plot the forest plot
    """fig = make_forestplot(summary_models, estimate="Hazard Ratio",
                        ll="Lower CI",
                        hl="Upper CI",
                        varlabel="Feature",
                        pval="p-value",
                        ylabel="Confidence interval",  # y-label title
                        xlabel="Hazard ratio",  # x-label title
                        labelpad=-350
                        )"""
    fig = simple_forestplot(summary_models, est_col="Hazard Ratio",
                        ll_col="Lower CI",
                        hl_col="Upper CI",
                        feature_col="Feature",
                        pval_col="p-value",
                        xlabel="Hazard ratio",  # x-label title
                            )

    plt.savefig("univariate_cox_models.svg", bbox_inches='tight')
    
    
    return p_values

def plot_kaplan_meier_subplots(
    data, time_col, event_col, feature_cols, fig_title, plot_names, ncols=3, out_path=None, group_labels_dict=None
):
    import math
    n_plots = len(feature_cols)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, (feature_col, plot_name) in enumerate(zip(feature_cols, plot_names)):
        ax = axes[idx]
        kmf = KaplanMeierFitter()
        """
        for value in [0, 1]:
            subset = data[data[feature_col] == value]
            if len(subset) == 0:
                continue
            kmf.fit(subset[time_col], event_observed=subset[event_col],
                    label="High" if value == 1 else "Low")

            kmf.plot_survival_function(ci_show=True, ax=ax)
        """
        for value in [0, 1]:
            subset = data[data[feature_col] == value]
            if len(subset) == 0:
                continue
            # Use custom group labels if provided
            if group_labels_dict and feature_col in group_labels_dict:
                group_label = group_labels_dict[feature_col][value]
            else:
                group_label = "High" if value == 1 else "Low"
            kmf.fit(subset[time_col], event_observed=subset[event_col],
                    label=group_label)
            kmf.plot_survival_function(ci_show=True, ax=ax)
        fraction_positive = data[feature_col].mean()
        fraction_text = f"Frac. high: {fraction_positive:.2f}"
        ax.text(0.5, 0.1, fraction_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5))
        ax.set_title(plot_name)
        ax.set_xlabel("Overall survival time (Years)")
        ax.set_ylabel("Survival probability")
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 5)
        ax.legend()
    
    # Hide any unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    #plt.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.4)
    if out_path:
        plt.savefig(out_path)
    #plt.show()


def plot_model_kaplan_meier(data, time_col, event_col, model_pred_cols, fig_title, out_path=None):
    import math
    n_plots = len(model_pred_cols)
    ncols = 3
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    print(model_pred_cols)
    for idx, col in enumerate(model_pred_cols):
        ax = axes[idx]
        # Use pre-computed dichotomized column if available, otherwise dichotomize at 0.5 or median
        if set(data[col].unique()) <= {0, 1}:
            bin_col = col
        else:
            cutoff = 0.5  # or np.median(data[col])
            bin_col = f"{col}_bin"
            data[bin_col] = (data[col] > cutoff).astype(int)
        kmf = KaplanMeierFitter()
        for value in [0, 1]:
            subset = data[data[bin_col] == value]
            if len(subset) == 0:
                continue
            kmf.fit(subset[time_col], event_observed=subset[event_col],
                    label="High" if value == 1 else "Low")
            kmf.plot_survival_function(ci_show=True, ax=ax)
        fraction_positive = data[bin_col].mean()
        fraction_text = f"Frac. high: {fraction_positive:.2f}"
        ax.text(0.5, 0.1, fraction_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5))
        ax.set_title(col)
        ax.set_xlabel("Overall survival time (Years)")
        ax.set_ylabel("Survival probability")
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 5)
        ax.legend()
        
        
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    #plt.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.4)
    if out_path:
        plt.savefig(out_path)
    #plt.show()


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
        #print(experiment_name)
        preds = pd.read_csv(os.path.join("./preds/", experiment_name + "_predictions.csv"))
        logits = pd.read_csv(os.path.join("./preds/", experiment_name + "_logits.csv"))
        preds = preds.melt(var_name="ID", value_name="pred")
        logits = logits.melt(var_name="ID", value_name="logit")
        preds.columns = ["ID", experiment_name + " pred"]
        logits.columns = ["ID", experiment_name + " logit"]
        patient_preds = pd.merge(logits, preds.drop(columns=["ID"]), left_index=True, right_index=True)#, on="ID")
        all_model_preds.append(patient_preds)
        #print(patient_preds)
    #print("merging")
    
    model_pred1 = all_model_preds[0]
    all_model_preds = all_model_preds[1:]
    for model_pred in all_model_preds:
        model_pred1 = pd.merge(model_pred1, model_pred.drop(columns=["ID"]), left_index=True, right_index=True)

    all_model_preds = model_pred1
    #all_model_preds = reduce(lambda left, right: pd.merge(left, right, on='ID', how="outer"), all_model_preds)
    #print("merge done")
    return names, all_model_preds


def get_patient_pred():
    names = [x.replace("_predictions.csv", "") for x in os.listdir("./preds/") if x .endswith("_predictions.csv")]
    all_model_preds = []
    for experiment_name in names:
        pred_logits = pd.read_csv(os.path.join("./preds/", experiment_name + "preds_logits.csv"))
        all_model_preds.append(preds_logits)
                                  
print("get patient predictions")
experiment_names, all_models_preds = get_patient_preds()#get_extended_preds()##

        
patient_data = pd.read_csv("patient_densities_morphologies.csv")
patient_data.rename(columns={"Gender": "Sex (Male)"}, inplace=True)
patient_data['ID'] = patient_data['ID'].astype(float)
print(all_models_preds)
all_models_preds['ID'] = all_models_preds['ID'].astype(float)

patient_data = dichotomize_clinpars(patient_data)

patient_data = cencor_after5years(patient_data)

patient_data_model_preds = pd.merge(patient_data, all_models_preds, on="ID")
#print(patient_data_model_preds)
#print(patient_data_model_preds[["ID", "label", "clinical parameters logit", "clinical parameters pred"]])

patient_data = patient_data_model_preds

model_pred_cols = [col for col in patient_data.columns if col.endswith('pred')]
# Rename columns: underscores to "+"
for col in model_pred_cols:
    new_col = col.replace("_", "+")
    new_col = new_col +  "."
    if new_col != col:
        patient_data.rename(columns={col: new_col}, inplace=True)


clinical_features = ['Age', 'Performance status', 'Stage', 'Sex (Male)', 'Smoking', 'LUAD']  # Edit to match your columns

density_features = [
    "Stroma CD4_Single",
    "Stroma CD4_Treg",
    "Stroma CD8_Single",
    "Stroma CD8_Treg",
    "Stroma B_cells",
    "Tumor CD4_Single",
    "Tumor CD4_Treg",
    "Tumor CD8_Single",
    "Tumor CD8_Treg",
    "Tumor B_cells"
]

density_rename_map = {}
for col in density_features:
    new_col = (
        col.replace("_", " ")
           .replace("Single", "eff.")
           #.replace("Treg", "T-reg")
    )
    density_rename_map[col] = new_col
    if col in patient_data.columns:
        patient_data.rename(columns={col: new_col}, inplace=True)

# Now update your list to reflect new column names
density_features = [density_rename_map[col] for col in density_features]


morphology_features = ['Nucleus Area', 'Nucleus Compactness', 'Nucleus Axis Ratio']  # Replace with actual names




clinical_titles = ["Age", "Performance status", "Stage", "Sex (Male)", "Smoking", "LUAD"]
density_titles = density_features  # or custom short names
morph_titles = ["Nucleus area", "Nucleus compactness", "Nucleus axis ratio"]



# For models, assuming your prediction columns are named as e.g. 'clinical parameters pred', 'densities pred', etc.
model_pred_cols = [col for col in patient_data.columns if col.endswith('pred.')]
model_logit_cols = [col for col in patient_data.columns if col.endswith('logit')]


patient_data[model_pred_cols] = patient_data[model_pred_cols].map(lambda x: 1 if x >= 0.5 else 0)
#patient_data[model_logit_cols] = patient_data[model_logit_cols].map(lambda x: 1 if x >= 0.5 else 0) 


#print(patient_data["densities pred"].values)
cutoffs = get_all_best_cutoffs(patient_data, experiment_names)
for feature in tqdm(cutoffs):
    cutoff = cutoffs[feature]
    patient_data[feature] = (patient_data[feature] > cutoff).astype(int)
#print(patient_data["densities pred"].values)




#print("accuracy:", accuracy_score(patient_data_model_preds["label"], patient_data_model_preds["morphologies pred"]))
print("accuracy:", roc_auc_score(patient_data_model_preds["label"], patient_data_model_preds["clinical parameters pred."]))
print(patient_data_model_preds.columns.values)
p_values = univariate_coxregression(patient_data)
multivariate_coxregression(patient_data, p_values)
cph, features, summary = stepwise_coxregression(patient_data, p_values)
cph, features, summary = forward_stepwise_coxregression(patient_data, p_values)

clinical_group_labels = {
    "Age":           {0: "<70",         1: "70+"},
    "Stage":         {0: "Stage I",     1: "Stage II–IV"},
    "Performance status": {0: "0",      1: "1–2"},
    "Sex (Male)":    {0: "Female",      1: "Male"},
    "Smoking":       {0: "Never",       1: "Ever"},
    "LUAD":          {0: "Other/SqCC",  1: "LUAD"},
}

"""for feature in cutoffs:
    plot_kaplan_meier(patient_data, "time", "event", feature, feature + ".png")

for feature in ['LUAD', 'Age', 'Gender', 'Smoking', 'Stage', 'Performance status']:
    plot_kaplan_meier(patient_data, "time", "event", feature, feature + ".png")
"""
#for name in experiment_names:
#    plot_kaplan_meier(patient_data, "time", "event", name + " pred", name + ".png")

# Group names for plot titles (shorter is better in subplot titles)

print("Columns in patient_data:")
print(patient_data.columns.tolist())


patient_data["time"] = patient_data["time"]/365 # days to years
plot_kaplan_meier_subplots(
    patient_data, "time", "event", clinical_features, "Kaplan-Meier: Clinical Parameters",
    clinical_titles, ncols=3, out_path="km_clinical.svg", group_labels_dict=clinical_group_labels
)

plot_kaplan_meier_subplots(
    patient_data, "time", "event", density_features, "Kaplan-Meier: Cell Densities",
    density_titles, ncols=3, out_path="km_densities.svg"
)

plot_kaplan_meier_subplots(
    patient_data, "time", "event", morphology_features, "Kaplan-Meier: Morphologies",
    morph_titles, ncols=3, out_path="km_morphologies.svg"
)

model_pred_cols = ['clinical parameters pred.', 'densities pred.', 'morphologies pred.', 'clinical parameters+densities pred.', 'clinical parameters+morphologies pred.', 'morphologies+densities pred.', 'clinical parameters+morphologies+densities pred.']

plot_model_kaplan_meier(
    patient_data, "time", "event", model_pred_cols, "Kaplan-Meier: Model Predictions",
    out_path="km_models.svg"
)

main_model_pred_cols = ['clinical parameters pred.', 'densities pred.', 'clinical parameters+densities pred.']

plot_model_kaplan_meier(
    patient_data, "time", "event", main_model_pred_cols, "Kaplan-Meier: Model Predictions",
    out_path="km_main_models.svg"
)
