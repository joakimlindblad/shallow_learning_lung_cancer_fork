
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def clean_density_name(name):
    if not isinstance(name, str):
        return name
    name = name.replace("_", " ")
    if name == "Gender":
        name = "Sex (Male)"
    return name

# Directory with results
plot_dir = './plots'

# Patterns for relevant files
weight_pattern = re.compile(r'(.+)_LogReg_feature_importance\.csv$')
perm_pattern = re.compile(r'(.+)_LogReg_perm_importance\.csv$')

# Human-readable names for experiments
exp_fullname_map = {
    "clinical parameters": "Clinical Parameters",
    "densities": "Immune Cell Densities",
    "morphologies": "Morphologies",
    "clinical parameters_densities": "Clinical Parameters + Densities",
    "clinical parameters_morphologies": "Clinical Parameters + Morphologies",
    "morphologies_densities": "Densities + Morphologies",
    "clinical parameters_morphologies_densities": "Clinical Parameters + Densities + Morphologies"
}

# Consistent experiment order for subplots
desired_order = [
    "clinical parameters",
    "densities",
    "morphologies",
    "clinical parameters_densities",
    "clinical parameters_morphologies",
    "morphologies_densities",
    "clinical parameters_morphologies_densities"
]

# Discover available files
weights = {}
perms = {}
for fname in os.listdir(plot_dir):
    w_match = weight_pattern.match(fname)
    p_match = perm_pattern.match(fname)
    if w_match:
        exp = w_match.group(1)
        weights[exp] = os.path.join(plot_dir, fname)
    elif p_match:
        exp = p_match.group(1)
        perms[exp] = os.path.join(plot_dir, fname)

# Only keep experiments where both weights and importances are available, and sort
exps = [e for e in desired_order if (e in weights and e in perms)]

# Load data for each experiment
weight_data = {}
perm_data = {}
for exp in exps:
    df = pd.read_csv(weights[exp])
    features = np.array([clean_density_name(f) for f in df['feature'].values])
    weight_data[exp] = (features, df['importance'].values)
    perm_df = pd.read_csv(perms[exp])
    perm_features = np.array([clean_density_name(f) for f in perm_df['feature'].values])
    perm_data[exp] = (perm_features, perm_df['importance'].values)

def plot_grid(data_dict, xlim, title, out_file):
    n = len(data_dict)
    cols = 3
    rows = (n + cols - 1) // cols
    max_feats = max(len(features) for features, _ in data_dict.values())
    fig_height = max(4 * rows, 0.2 * max_feats * rows)
    fig = plt.figure(figsize=(cols * 4, fig_height))
    gs = GridSpec(rows, cols, figure=fig)
    for i, exp in enumerate(data_dict):
        features, importances = data_dict[exp]
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])
        idx = np.argsort(importances)
        ax.barh(features[idx], importances[idx], color='skyblue')
        ax.set_xlim(xlim)
        exp_title = exp_fullname_map.get(exp, exp)
        ax.set_title(exp_title, fontsize=12, x=0.4)
        ax.tick_params(axis='y', labelsize=8)
        ax.axvline(x=0, color='red', linestyle='--')
    # Empty subplots if n doesn't fill grid
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        fig.add_subplot(gs[r, c]).axis('off')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

def plot_single(features, importances, xlim, title, out_file):
    fig, ax = plt.subplots(figsize=(6, 0.4*len(features)+2))
    idx = np.argsort(importances)
    ax.barh(features[idx], importances[idx], color='skyblue')
    ax.set_xlim(xlim)
    #ax.set_title(title, fontsize=14)
    ax.tick_params(axis='y', labelsize=9)
    ax.axvline(x=0, color='red', linestyle='--')
    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

def plot_double(
    features1, importances1, xlim1, title1,
    features2, importances2, xlim2, title2,
    suptitle, out_file
):
    """
    Plot two horizontal bar plots side by side.
    Each plot is sorted independently by its own values (ascending, like plot_grid).
    """
    
    # Use the same approach as plot_grid
    N = max(len(features1), len(features2))
    fig_height = max(4, 0.2 * N)
    fig, axes = plt.subplots(1, 2, figsize=(8, fig_height))
    
    # Left plot - Feature Weights
    idx1 = np.argsort(importances1)
    axes[0].barh(features1[idx1], importances1[idx1], color='skyblue')
    axes[0].set_xlim(xlim1)
    axes[0].set_title(title1, fontsize=12)
    axes[0].tick_params(axis='y', labelsize=8)
    axes[0].axvline(x=0, color='red', linestyle='--')
    
    # Right plot - Permutation Importances
    idx2 = np.argsort(importances2)
    axes[1].barh(features2[idx2], importances2[idx2], color='skyblue')
    axes[1].set_xlim(xlim2)
    axes[1].set_title(title2, fontsize=12)
    axes[1].tick_params(axis='y', labelsize=8)
    axes[1].axvline(x=0, color='red', linestyle='--')
    
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    

    
    
# Grid of feature weights for all experiments (ordered, full titles)
plot_grid(
    {exp: weight_data[exp] for exp in exps},
    xlim=(-0.45, 0.45),
    title='Logistic Regression Feature Weights',
    out_file='feature_weights_grid.svg'
)

# Grid of permutation importances for all experiments (ordered, full titles)
plot_grid(
    {exp: perm_data[exp] for exp in exps},
    xlim=(-0.01, 0.065),
    title='Logistic Regression Permutation Importances',
    out_file='perm_importances_grid.svg'
)

# Standalone plots for Clinical Parameters + Densities + Morphologies
combo_exp = "clinical parameters_morphologies_densities"
if combo_exp in weight_data:
    plot_single(
        weight_data[combo_exp][0],
        weight_data[combo_exp][1],
        xlim=(-0.45, 0.45),
        title='Clinical Parameters + Densities + Morphologies',
        out_file='feature_weights_clinical_parameters_densities_morphologies.svg'
    )
if combo_exp in perm_data:
    plot_single(
        perm_data[combo_exp][0],
        perm_data[combo_exp][1],
        xlim=(-0.01, 0.065),
        title='Clinical Parameters + Densities + Morphologies',
        out_file='perm_importances_clinical_parameters_densities_morphologies.svg'
    )


combo_exp = "clinical parameters_morphologies_densities"
if combo_exp in weight_data and combo_exp in perm_data:
    plot_double(
        weight_data[combo_exp][0], weight_data[combo_exp][1], (-0.45, 0.45), "Model feature weights",
        perm_data[combo_exp][0], perm_data[combo_exp][1], (-0.01, 0.065), "Permutation feature importances",
        suptitle="Clinical Parameters + Densities + Morphologies",
        out_file="weights_perm_importance_combined.svg"
    )
