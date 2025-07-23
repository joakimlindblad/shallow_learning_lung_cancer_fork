import matplotlib.pyplot as plt
import numpy as np

def format_pval(p):
    """Format p-value for forest plot: ≤3 decimals, asterisks for significance."""
    try:
        p_float = float(p)
    except Exception:
        return str(p)
    if p_float > 0.05:
        return f"{p_float:.2f}".rstrip("0").rstrip(".")  # max 3 decimals, trim trailing
    elif p_float > 0.01:
        return f"{p_float:.2f}*".rstrip("0").rstrip(".")  # max 3 decimals + asterisk
    elif p_float > 0.001:
        return "**"
    else:
        return "***"

    
def simple_forestplot(
    df,
    feature_col="Feature",
    est_col="Hazard Ratio",
    ll_col="Lower CI",
    hl_col="Upper CI",
    pval_col="p-value",
    ci_digits=2,
    fontfamily="DejaVu Sans",
    fontsize=15,
    xlim=None,
    xlabel="Hazard ratio",
    hr_refline=1,
    hr_refline_color="red",
    hr_refline_style="--",
    dot_color="black",
    ci_color="black",
    savepath=None
):
    # --- Calculate scaling based on feature name length and number of rows ---
    longest_name = max([len(str(x)) for x in df[feature_col]])
    n_features = len(df)
    # Each character ~0.13 units of width; min 3.2, adjust for long names
    feat_col_width = max(3.2, 0.125 * longest_name + 1.5)
    ci_col_width = 1.7 if df[ll_col].dtype != object else 4
    pval_col_width = 1.5
    forest_col_width = 4.5
    width_ratios = [feat_col_width, ci_col_width, pval_col_width, forest_col_width]
    # Figure width: base width plus extra for long feature names
    fig_width = sum(width_ratios) + 1
    fig_height = max(1.3, 0.55 * n_features + 1.5)

    # Format CI and pvals
    ci_texts = df.apply(
        lambda row: f"{row[est_col]:.{ci_digits}f} ({row[ll_col]:.{ci_digits}f}-{row[hl_col]:.{ci_digits}f})", axis=1
    )
    pval_texts = df[pval_col].apply(format_pval)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, 4, width_ratios=width_ratios, wspace=0.2)
    ax_feat = fig.add_subplot(gs[0,0])
    ax_ci   = fig.add_subplot(gs[0,1], sharey=ax_feat)
    ax_pval = fig.add_subplot(gs[0,2], sharey=ax_feat)
    ax_forest = fig.add_subplot(gs[0,3], sharey=ax_feat)

    yticks = np.arange(n_features)

    # Remove ticks/grid for text axes
    for ax in [ax_feat, ax_ci, ax_pval]:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.tick_params(left=False, right=False, top=False, bottom=False)
        ax.spines[:].set_visible(False)

    ax_feat.set_ylim(-0.5, n_features-0.5)
    ax_feat.set_xlim(0, 1)
    ax_feat.invert_yaxis()
    #ax_feat.set_title(feature_col, fontsize=fontsize, pad=12, fontfamily=fontfamily)
    for y, val in enumerate(df[feature_col]):
        ax_feat.text(0, y, str(val), fontfamily=fontfamily, fontsize=fontsize, va="center", ha="left")

    ax_ci.set_ylim(ax_feat.get_ylim())
    ax_ci.set_xlim(0, 1)
    ax_ci.set_title("Confidence Interval", fontsize=fontsize, pad=12, fontfamily=fontfamily, fontweight="bold")
    for y, val in enumerate(ci_texts):
        ax_ci.text(0.5, y, val, fontfamily=fontfamily, fontsize=fontsize, va="center", ha="center")

    ax_pval.set_ylim(ax_feat.get_ylim())
    ax_pval.set_xlim(0, 1)
    ax_pval.set_title("p-value", fontsize=fontsize, pad=12, fontfamily=fontfamily, fontweight="bold", loc="right")
    for y, val in enumerate(pval_texts):
        ax_pval.text(1, y, val, fontfamily=fontfamily, fontsize=fontsize, va="center", ha="right")

    # Forest plot
    ax_forest.set_ylim(ax_feat.get_ylim())
    ax_forest.set_yticks(yticks)
    ax_forest.set_yticklabels([])
    for y, (_, row) in enumerate(df.iterrows()):
        est = row[est_col]
        ll = row[ll_col]
        hl = row[hl_col]
        ax_forest.plot([ll, hl], [y, y], color=ci_color, lw=2.3, zorder=1)
        ax_forest.plot(est, y, "o", color=dot_color, markersize=9, zorder=2)
    ax_forest.axvline(hr_refline, color=hr_refline_color, ls=hr_refline_style, lw=2, zorder=0)
    ax_forest.tick_params(axis="y", left=False, right=False, labelleft=False)
    ax_forest.tick_params(axis="x", labelsize=fontsize-2)
    ax_forest.set_xlabel(xlabel, fontsize=fontsize+1, fontfamily=fontfamily)
    if xlim:
        ax_forest.set_xlim(*xlim)
    else:
        min_ll, max_hl = df[ll_col].min(), df[hl_col].max()
        ax_forest.set_xlim(max(0, min_ll-0.1), max_hl+0.1)
    ax_forest.spines["left"].set_visible(False)
    ax_forest.spines["right"].set_visible(False)
    ax_forest.grid(axis="x", linestyle=":", alpha=0.7)

    fig.align_ylabels([ax_feat, ax_ci, ax_pval, ax_forest])

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig

# Usage:
# fig = simple_forestplot(df)
# plt.show()
