import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

# ============================================================
#                   CONFIGURATION
# ============================================================

# Optimizer set used in analysis/statistical comparisons.
MANUAL_SEEDS = [21, 31, 42, 48, 64, 123, 256, 2048, 3651]
OPTIMIZERS = [
    "Rprop", "RAdam", "AdamW", "Adam", "Adamax", "NAdam",
    "SGD", "AdaGrad", "RMSProp", "Adadelta", "ASGD"
]
N_FOLDS = 5
INITIAL_POINTS = 5
BO_ITERATIONS = 20
EPOCHS_PER_TRIAL = 20

USE_GENERATED_DATA = False  # Set to False once you have real results

SIGNIFICANCE_LEVEL = 0.05

# ============================================================
#              SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_summary(seeds, optimizers, rng_seed=0):
    """
    Generate a fake bo_summary.csv-style DataFrame for testing the analysis
    pipeline. Each optimizer gets a base accuracy drawn from a plausible range
    and per seed noise is added so that the ranking is not perfectly constant.
    """
    rng = np.random.RandomState(rng_seed)

    # Plausible base accuracies (mean across seeds) for each optimizer.
    # Deliberately make some close and some clearly different.
    base_acc = {
        "SGD":      0.870,
        "Adam":     0.885,
        "RMSProp":  0.880,
        "AdaGrad":  0.855,
    }

    rows = []
    for seed in seeds:
        for opt in optimizers:
            noise = rng.normal(0, 0.008)  # per-seed noise
            acc = np.clip(base_acc.get(opt, 0.86) + noise, 0.0, 1.0)
            fold_std = rng.uniform(0.003, 0.010)
            lr = 10 ** rng.uniform(-4, -1)
            rows.append({
                "seed": seed,
                "optimizer": opt,
                "best_accuracy": round(acc, 6),
                "accuracy_std": round(fold_std, 6),
                "best_learning_rate": round(lr, 6),
                "best_final_loss": round(rng.uniform(0.3, 0.6), 6),
                "total_trials": INITIAL_POINTS + BO_ITERATIONS,
            })
    return pd.DataFrame(rows)


def generate_synthetic_trials(seeds, optimizers, rng_seed=0):
    """
    Generate a fake bo_trials.csv-style DataFrame with per-trial data
    including convergence curves, for testing plotting functions.
    """
    rng = np.random.RandomState(rng_seed)
    total_trials = INITIAL_POINTS + BO_ITERATIONS

    base_acc = {"SGD": 0.870, "Adam": 0.885, "RMSProp": 0.880, "AdaGrad": 0.855}

    rows = []
    for seed in seeds:
        for opt in optimizers:
            best_so_far = 0.0
            for i in range(total_trials):
                lr = 10 ** rng.uniform(-4, -1)
                noise = rng.normal(0, 0.012)
                acc = np.clip(base_acc.get(opt, 0.86) + noise, 0.0, 1.0)
                best_so_far = max(best_so_far, acc)

                fold_accs = [np.clip(acc + rng.normal(0, 0.005), 0, 1) for _ in range(N_FOLDS)]

                # Simulated convergence: loss decreases, val_acc increases
                epoch_losses = []
                epoch_val_accs = []
                loss = rng.uniform(1.5, 2.5)
                for e in range(EPOCHS_PER_TRIAL):
                    loss *= rng.uniform(0.88, 0.98)
                    epoch_losses.append(round(loss, 6))
                    epoch_val_accs.append(round(np.clip(acc - 0.05 + 0.05 * (e / EPOCHS_PER_TRIAL) + rng.normal(0, 0.003), 0, 1), 6))

                row = {
                    "seed": seed,
                    "optimizer": opt,
                    "iteration": i,
                    "trial_type": "initial" if i < INITIAL_POINTS else "bo",
                    "learning_rate": round(lr, 6),
                    "mean_accuracy": round(acc, 6),
                    "accuracy_std": round(np.std(fold_accs), 6),
                    "best_accuracy_so_far": round(best_so_far, 6),
                    "ei_value": round(rng.uniform(0, 0.01), 6) if i >= INITIAL_POINTS else None,
                }

                # Add fold accuracies
                for f_idx, fa in enumerate(fold_accs):
                    row[f"fold_{f_idx+1}_accuracy"] = round(fa, 6)

                # Add epoch-level data
                for e_idx in range(EPOCHS_PER_TRIAL):
                    row[f"epoch_{e_idx+1}_avg_loss"] = epoch_losses[e_idx]
                    row[f"epoch_{e_idx+1}_avg_val_acc"] = epoch_val_accs[e_idx]

                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
#              LOAD DATA
# ============================================================

def load_trials(path="results/bo_trials.csv"):
    """Load trials CSV, skipping comment lines starting with #."""
    return pd.read_csv(path, comment="#")


def build_summary_from_trials(trials_df):
    """
    Derive a bo_summary-style DataFrame from bo_trials.csv.
    For each (seed, optimizer), pick the trial with the highest mean_accuracy.
    """
    fold_acc_cols = [c for c in trials_df.columns if c.startswith("fold_") and c.endswith("_accuracy")]

    rows = []
    for (seed, opt), group in trials_df.groupby(["seed", "optimizer"]):
        best_idx = group["mean_accuracy"].idxmax()
        best_row = group.loc[best_idx]

        # Compute accuracy_std from fold columns if available
        if fold_acc_cols:
            fold_vals = best_row[fold_acc_cols].dropna().values.astype(float)
            acc_std = float(np.std(fold_vals)) if len(fold_vals) > 0 else 0.0
        else:
            acc_std = float(best_row.get("accuracy_std", 0.0))

        # Get final epoch loss (last non-NaN epoch_*_avg_loss column)
        epoch_loss_cols = sorted([c for c in trials_df.columns if c.startswith("epoch_") and c.endswith("_avg_loss")])
        final_loss = np.nan
        if epoch_loss_cols:
            loss_vals = best_row[epoch_loss_cols].dropna().values.astype(float)
            if len(loss_vals) > 0:
                final_loss = loss_vals[-1]

        rows.append({
            "seed": seed,
            "optimizer": opt,
            "best_accuracy": float(best_row["mean_accuracy"]),
            "accuracy_std": acc_std,
            "best_learning_rate": float(best_row["learning_rate"]),
            "best_final_loss": final_loss,
            "total_trials": len(group),
        })

    return pd.DataFrame(rows)


# ============================================================
#              STATISTICAL TESTS
# ============================================================

def build_accuracy_matrix(summary_df):
    """
    Pivot the summary into a (seeds × optimizers) matrix of best accuracies.
    Rows = seeds (blocks), Columns = optimizers (treatments).
    """
    pivot = summary_df.pivot(index="seed", columns="optimizer", values="best_accuracy")
    # Reorder columns to match OPTIMIZERS list (intersection only)
    cols = [o for o in OPTIMIZERS if o in pivot.columns]
    return pivot[cols]


def run_friedman_test(acc_matrix):
    """
    Friedman test: are the optimizers significantly different?
    Input: DataFrame with shape (n_seeds, n_optimizers).
    Returns: (statistic, p_value).
    """
    # Each column is one "treatment" (optimizer), each row is one "block" (seed).
    groups = [acc_matrix[col].values for col in acc_matrix.columns]
    stat, p = friedmanchisquare(*groups)
    return stat, p


def run_posthoc_wilcoxon(acc_matrix, alpha=SIGNIFICANCE_LEVEL):
    """
    Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni step-down correction.
    Returns a DataFrame with columns: opt_A, opt_B, statistic, p_raw, p_corrected, significant.
    """
    optimizers = list(acc_matrix.columns)
    pairs = list(combinations(optimizers, 2))
    m = len(pairs)
    raw_results = []

    for opt_a, opt_b in pairs:
        a = acc_matrix[opt_a].values
        b = acc_matrix[opt_b].values

        diff = a - b
        if np.all(diff == 0):
            stat, p_raw = np.nan, 1.0
        else:
            stat, p_raw = wilcoxon(a, b, alternative="two-sided")

        raw_results.append((opt_a, opt_b, stat, p_raw))

    # Sort by raw p-value for Holm step-down
    raw_results.sort(key=lambda x: x[3])

    results = []
    holm_rejected = True  # Stays True until first non-rejection
    prev_p_corrected = 0.0

    for i, (opt_a, opt_b, stat, p_raw) in enumerate(raw_results):
        # Holm multiplier: (m - i) where i is 0-indexed
        p_corrected = p_raw * (m - i)
        # Ensure monotonicity: corrected p can't decrease
        p_corrected = max(p_corrected, prev_p_corrected)
        p_corrected = min(p_corrected, 1.0)
        prev_p_corrected = p_corrected

        # Holm step-down: once we fail to reject, all subsequent are not rejected
        if holm_rejected and p_corrected < alpha:
            significant = True
        else:
            holm_rejected = False
            significant = False

        results.append({
            "opt_A": opt_a,
            "opt_B": opt_b,
            "statistic": stat,
            "p_raw": p_raw,
            "p_corrected": p_corrected,
            "significant": significant,
        })

    return pd.DataFrame(results)


def compute_mean_ranks(acc_matrix, optimizer_order=None):
    """
    Compute mean rank of each optimizer across seeds (lower = better accuracy).
    Friedman test is based on these ranks internally.
    """
    if optimizer_order is not None:
        available = [opt for opt in optimizer_order if opt in acc_matrix.columns]
        if available:
            acc_matrix = acc_matrix[available]

    # rank along columns (optimizers) per row (seed); higher accuracy → rank 1
    ranks = acc_matrix.rank(axis=1, ascending=False)
    return ranks.mean().sort_values()


def compute_kendalls_w(acc_matrix):
    """
    Kendall's W (coefficient of concordance) derived from Friedman's χ²:
        W = χ²_F / (n * (k - 1))

    Measures agreement between seeds on the ranking of optimizers.
    W ∈ [0, 1]: 0 = no agreement, 1 = perfect agreement.
    """
    stat, _ = run_friedman_test(acc_matrix)
    n = acc_matrix.shape[0]  # number of seeds (judges)
    k = acc_matrix.shape[1]  # number of optimizers (objects)

    w = stat / (n * (k - 1))

    # Interpret
    if w < 0.1:
        interpretation = "negligible agreement"
    elif w < 0.3:
        interpretation = "weak agreement"
    elif w < 0.5:
        interpretation = "moderate agreement"
    elif w < 0.7:
        interpretation = "strong agreement"
    else:
        interpretation = "very strong agreement"

    return w, interpretation


# ============================================================
#              VISUALISATIONS
# ============================================================

def get_optimizer_color_map(optimizers):
    """Return a consistent optimizer -> color mapping."""
    unique_opts = sorted(pd.unique(list(optimizers)))
    cmap = plt.colormaps.get_cmap("tab20")
    colors = [cmap(i / max(len(unique_opts), 1)) for i in range(len(unique_opts))]
    return {opt: color for opt, color in zip(unique_opts, colors)}

def plot_accuracy_boxplot(acc_matrix, save_path="results/accuracy_boxplot.png"):
    """Box plot of best accuracy per optimizer across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    melted = acc_matrix.reset_index().melt(id_vars="seed", var_name="Optimizer", value_name="Best Accuracy")
    order = list(acc_matrix.columns)
    color_map = get_optimizer_color_map(order)
    sns.boxplot(data=melted, x="Optimizer", y="Best Accuracy", ax=ax, order=order, palette=color_map)
    sns.stripplot(data=melted, x="Optimizer", y="Best Accuracy", ax=ax, color="black", size=4, alpha=0.6)
    ax.set_title("Best Accuracy per Optimizer (across seeds)")
    ax.set_ylabel("Best Cross-Validated Accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mean_rank_bar(acc_matrix, save_path="results/mean_rank.png", optimizer_order=None):
    """Bar chart of mean ranks (Friedman-style)."""
    if optimizer_order is None:
        optimizer_order = OPTIMIZERS

    available = [opt for opt in optimizer_order if opt in acc_matrix.columns]
    if len(available) < 2:
        raise ValueError("Need at least 2 optimizers in acc_matrix to compute mean ranks.")

    mean_ranks = compute_mean_ranks(acc_matrix[available], optimizer_order=available)
    fig_height = max(5, 0.45 * len(mean_ranks) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    color_map = get_optimizer_color_map(mean_ranks.index)
    bar_colors = [color_map[opt] for opt in mean_ranks.index]
    mean_ranks.plot.barh(ax=ax, color=bar_colors)
    ax.set_xlabel("Mean Rank (lower = better)")
    ax.set_title("Mean Rank of Optimizers (Friedman)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bo_convergence(trials_df, save_path="results/bo_convergence.png"):
    """Plot best-accuracy-so-far over BO iterations, averaged across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = get_optimizer_color_map(trials_df["optimizer"].unique())

    for opt in sorted(trials_df["optimizer"].unique()):
        opt_df = trials_df[trials_df["optimizer"] == opt]
        # Average best_accuracy_so_far across seeds per iteration
        convergence = opt_df.groupby("iteration")["best_accuracy_so_far"].mean()
        ax.plot(convergence.index, convergence.values, marker="o", markersize=3,
                label=opt, color=color_map[opt])

    ax.axvline(x=INITIAL_POINTS - 0.5, color="gray", linestyle="--", alpha=0.5, label="BO starts")
    ax.set_xlabel("Trial Iteration")
    ax.set_ylabel("Best Accuracy So Far (mean across seeds)")
    ax.set_title("Bayesian Optimization Convergence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_lr_vs_accuracy(trials_df, save_path="results/lr_vs_accuracy.png"):
    """Scatter plot of learning rate vs accuracy for all trials, coloured by optimizer,
    with a LOWESS trend line per optimizer for readability."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, ax = plt.subplots(figsize=(12, 7))
    color_map = get_optimizer_color_map(trials_df["optimizer"].unique())

    for opt in sorted(trials_df["optimizer"].unique()):
        subset = trials_df[trials_df["optimizer"] == opt].copy()
        color = color_map[opt]

        # Use log-transformed LR for scatter + LOWESS so spacing is even
        log_lr = np.log10(subset["learning_rate"].values)

        # Scatter (small, semi-transparent)
        ax.scatter(subset["learning_rate"], subset["mean_accuracy"],
                   alpha=0.25, s=10, color=color, edgecolors="none")

        # LOWESS trend line
        if len(subset) >= 5:
            smoothed = lowess(subset["mean_accuracy"].values, log_lr, frac=0.3, is_sorted=False)
            sorted_idx = np.argsort(smoothed[:, 0])
            ax.plot(10 ** smoothed[sorted_idx, 0], smoothed[sorted_idx, 1],
                    color=color, linewidth=2, label=opt)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_title("Learning Rate vs Accuracy (all trials)")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_lr_vs_loss(trials_df, save_path="results/lr_vs_loss.png"):
    """Scatter plot of learning rate vs final training loss (log scale on both axes),
    with a LOWESS trend line per optimizer."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # Compute final epoch loss for each trial
    epoch_loss_cols = sorted([c for c in trials_df.columns
                              if c.startswith("epoch_") and c.endswith("_avg_loss")])
    if not epoch_loss_cols:
        print("  Skipping LR vs Loss — no epoch loss columns found.")
        return

    df = trials_df.copy()
    # Final loss = last non-NaN epoch loss per trial
    df["final_loss"] = df[epoch_loss_cols].apply(
        lambda row: row.dropna().values[-1] if len(row.dropna()) > 0 else np.nan, axis=1
    )
    df = df.dropna(subset=["final_loss"])
    df = df[df["final_loss"] > 0]  # needed for log scale

    fig, ax = plt.subplots(figsize=(12, 7))
    color_map = get_optimizer_color_map(df["optimizer"].unique())

    for opt in sorted(df["optimizer"].unique()):
        subset = df[df["optimizer"] == opt]
        color = color_map[opt]
        log_lr = np.log10(subset["learning_rate"].values)

        ax.scatter(subset["learning_rate"], subset["final_loss"],
                   alpha=0.25, s=10, color=color, edgecolors="none")

        if len(subset) >= 5:
            smoothed = lowess(np.log10(subset["final_loss"].values), log_lr,
                              frac=0.3, is_sorted=False)
            sorted_idx = np.argsort(smoothed[:, 0])
            ax.plot(10 ** smoothed[sorted_idx, 0], 10 ** smoothed[sorted_idx, 1],
                    color=color, linewidth=2, label=opt)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Final Training Loss (log scale)")
    ax.set_title("Learning Rate vs Final Training Loss (all trials)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(trials_df, save_path="results/training_curves.png"):
    """
    For each optimizer, plot the average training loss curve of the best trial
    (highest mean_accuracy) averaged across seeds.
    """
    epoch_loss_cols = [f"epoch_{e+1}_avg_loss" for e in range(EPOCHS_PER_TRIAL)]
    # Only keep columns that exist
    epoch_loss_cols = [c for c in epoch_loss_cols if c in trials_df.columns]

    if not epoch_loss_cols:
        print("  Skipping training curves — no epoch loss columns found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = get_optimizer_color_map(trials_df["optimizer"].unique())

    for opt in sorted(trials_df["optimizer"].unique()):
        opt_df = trials_df[trials_df["optimizer"] == opt]
        # Pick the best trial per seed, then average the loss curves
        best_per_seed = opt_df.loc[opt_df.groupby("seed")["mean_accuracy"].idxmax()]
        avg_curve = best_per_seed[epoch_loss_cols].mean(axis=0).values
        ax.plot(range(1, len(avg_curve) + 1), avg_curve, label=opt, color=color_map[opt])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Training Loss")
    ax.set_title("Training Loss Curve (best trial per seed, averaged)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ei_over_iterations(trials_df, save_path="results/ei_over_iterations.png"):
    """Plot Expected Improvement values over BO iterations, averaged across seeds."""
    bo_trials = trials_df[trials_df["trial_type"] == "bo"].copy()
    if bo_trials.empty or "ei_value" not in bo_trials.columns:
        print("  Skipping EI plot — no BO trials or ei_value column found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = get_optimizer_color_map(bo_trials["optimizer"].unique())
    for opt in sorted(bo_trials["optimizer"].unique()):
        opt_df = bo_trials[bo_trials["optimizer"] == opt]
        ei_avg = opt_df.groupby("iteration")["ei_value"].mean()
        ax.plot(ei_avg.index, ei_avg.values, marker="o", markersize=3, label=opt, color=color_map[opt])

    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Expected Improvement (mean across seeds)")
    ax.set_title("Expected Improvement over Bayesian Optimization Iterations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_early_stopping_frequency(trials_df, save_path="results/early_stopping_freq.png"):
    """Bar chart showing fraction of folds that triggered early stopping, per optimizer."""
    es_cols = [c for c in trials_df.columns if c.endswith("_early_stopped")]
    if not es_cols:
        print("  Skipping early stopping plot — no early_stopped columns found.")
        return

    df = trials_df.copy()
    df["es_rate"] = df[es_cols].mean(axis=1)  # fraction of folds that early-stopped per trial

    fig, ax = plt.subplots(figsize=(10, 6))
    opt_es = df.groupby("optimizer")["es_rate"].mean().sort_values(ascending=False)
    color_map = get_optimizer_color_map(opt_es.index)
    colors = [color_map[opt] for opt in opt_es.index]
    opt_es.plot.bar(ax=ax, color=colors)
    ax.set_ylabel("Early Stopping Rate (fraction of folds)")
    ax.set_xlabel("Optimizer")
    ax.set_title("Early Stopping Frequency per Optimizer (across all trials & seeds)")

    max_rate = float(opt_es.max()) if len(opt_es) else 0.0
    if max_rate <= 0:
        y_max = 0.1
    else:
        y_max = min(1.0, max_rate * 1.25 + 0.02)

    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3, axis="y")
    label_offset = max(0.01, y_max * 0.02)
    for i, (opt, val) in enumerate(opt_es.items()):
        ax.text(i, min(y_max, val + label_offset), f"{val:.1%}", ha="center", fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fold_variance_boxplot(trials_df, save_path="results/fold_variance_boxplot.png"):
    """Box plot of accuracy_std (cross-fold variance) per optimizer."""
    if "accuracy_std" not in trials_df.columns:
        print("  Skipping fold variance plot — no accuracy_std column found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    melted = trials_df[["optimizer", "accuracy_std"]].copy()
    order = melted.groupby("optimizer")["accuracy_std"].median().sort_values().index.tolist()
    color_map = get_optimizer_color_map(order)
    sns.boxplot(data=melted, x="optimizer", y="accuracy_std", ax=ax, palette=color_map, order=order)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Accuracy Std (across folds)")
    ax.set_title("Cross-Fold Variance per Optimizer (lower = more stable)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_best_lr_distribution(summary_df, save_path="results/best_lr_distribution.png"):
    """Violin plot of the best learning rate found per optimizer across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = sorted(summary_df["optimizer"].unique())
    color_map = get_optimizer_color_map(order)
    sns.violinplot(data=summary_df, x="optimizer", y="best_learning_rate", ax=ax,
                   palette=color_map, order=order, inner="point", cut=0)
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Best Learning Rate (log scale)")
    ax.set_title("Distribution of Optimal Learning Rate per Optimizer (across seeds)")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_val_accuracy_curves(trials_df, save_path="results/val_accuracy_curves.png"):
    """
    For each optimizer, plot the average validation accuracy curve of the best trial
    (highest mean_accuracy) averaged across seeds.
    """
    epoch_val_cols = [c for c in sorted(trials_df.columns)
                      if c.startswith("epoch_") and c.endswith("_avg_val_acc")]
    if not epoch_val_cols:
        print("  Skipping validation accuracy curves — no epoch val_acc columns found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = get_optimizer_color_map(trials_df["optimizer"].unique())
    for opt in sorted(trials_df["optimizer"].unique()):
        opt_df = trials_df[trials_df["optimizer"] == opt]
        best_per_seed = opt_df.loc[opt_df.groupby("seed")["mean_accuracy"].idxmax()]
        avg_curve = best_per_seed[epoch_val_cols].mean(axis=0).values
        ax.plot(range(1, len(avg_curve) + 1), avg_curve, marker="o", markersize=3,
                label=opt, color=color_map[opt])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Validation Accuracy")
    ax.set_title("Validation Accuracy Curve (best trial per seed, averaged)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_wilcoxon_heatmap(posthoc_df, optimizers, save_path="results/wilcoxon_heatmap.png"):
    """Heatmap of pairwise corrected p-values from Wilcoxon tests."""
    n = len(optimizers)
    p_matrix = pd.DataFrame(np.ones((n, n)), index=optimizers, columns=optimizers)

    for _, row in posthoc_df.iterrows():
        p_matrix.loc[row["opt_A"], row["opt_B"]] = row["p_corrected"]
        p_matrix.loc[row["opt_B"], row["opt_A"]] = row["p_corrected"]

    # Diagonal = NaN for cleaner display
    np.fill_diagonal(p_matrix.values, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(p_matrix.astype(float), annot=True, fmt=".4f", cmap="RdYlGn",
                vmin=0, vmax=0.1, ax=ax, linewidths=0.5)
    ax.set_title(f"Pairwise Wilcoxon p-values (Bonferroni-corrected, α={SIGNIFICANCE_LEVEL})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
#              MAIN
# ============================================================

def print_section(title):
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def build_optimizer_accuracy_overview(summary_df):
    """Aggregate best_accuracy across seeds for each optimizer."""
    overview = summary_df.groupby("optimizer").agg(
        mean_acc=("best_accuracy", "mean"),
        std_acc=("best_accuracy", "std"),
        min_acc=("best_accuracy", "min"),
        max_acc=("best_accuracy", "max"),
        median_acc=("best_accuracy", "median"),
    ).sort_values("mean_acc", ascending=False)

    overview.index.name = "Optimizer"
    return overview


def build_optimizer_lr_overview(summary_df):
    """Aggregate best_learning_rate across seeds for each optimizer."""
    overview = summary_df.groupby("optimizer").agg(
        mean_lr=("best_learning_rate", "mean"),
        std_lr=("best_learning_rate", "std"),
        min_lr=("best_learning_rate", "min"),
        max_lr=("best_learning_rate", "max"),
        median_lr=("best_learning_rate", "median"),
    ).sort_values("mean_lr", ascending=False)

    overview.index.name = "Optimizer"
    return overview


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    # Pandas display options for clean terminal output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.6f}".format)

    # ── Load or generate data ─────────────────────────────────
    if USE_GENERATED_DATA:
        print("⚠  Using SYNTHETIC data for testing.")
        print("   Set USE_GENERATED_DATA = False once real results exist.\n")
        trials_df = generate_synthetic_trials(MANUAL_SEEDS, OPTIMIZERS)
        summary_df = build_summary_from_trials(trials_df)
    else:
        print("Loading real data from results/bo_trials.csv\n")
        trials_df = load_trials()
        summary_df = build_summary_from_trials(trials_df)

    # ── Data overview ─────────────────────────────────────────
    print_section("DATA OVERVIEW")
    print(f"  Trials loaded:     {len(trials_df)}")
    print(f"  Seeds:             {sorted(trials_df['seed'].unique())}")
    print(f"  Optimizers:        {sorted(trials_df['optimizer'].unique())}")
    print(f"  Trials per combo:  {trials_df.groupby(['seed', 'optimizer']).size().iloc[0]}")

    # ── Summary per (seed, optimizer): best trial ─────────────
    print_section("BEST TRIAL PER (SEED, OPTIMIZER)")
    summary_display = summary_df.copy()
    summary_display = summary_display.sort_values(["optimizer", "seed"])
    print(summary_display.to_string(index=False))

    # ── Optimizer accuracy overview (aggregated across seeds) ─
    print_section("OPTIMIZER ACCURACY OVERVIEW (aggregated across seeds)")
    accuracy_overview = build_optimizer_accuracy_overview(summary_df)
    print(accuracy_overview.to_string())

    # ── Optimizer LR overview (aggregated across seeds) ───────
    print_section("OPTIMIZER LEARNING-RATE OVERVIEW (aggregated across seeds)")
    lr_overview = build_optimizer_lr_overview(summary_df)
    print(lr_overview.to_string())

    # ── Accuracy matrix (seeds × optimizers) ──────────────────
    print_section("ACCURACY MATRIX (seeds × optimizers)")
    acc_matrix = build_accuracy_matrix(summary_df)
    print(acc_matrix.to_string())

    # ── Mean ranks ────────────────────────────────────────────
    print_section("MEAN RANKS (lower = better)")
    mean_ranks = compute_mean_ranks(acc_matrix)
    rank_df = pd.DataFrame({"Optimizer": mean_ranks.index, "Mean Rank": mean_ranks.values})
    print(rank_df.to_string(index=False))

    # ── Friedman test ─────────────────────────────────────────
    print_section("FRIEDMAN TEST")
    stat, p = run_friedman_test(acc_matrix)
    friedman_df = pd.DataFrame([{
        "Test": "Friedman χ²",
        "Statistic": stat,
        "p-value": f"{p:.2e}",
        "Significant": "YES" if p < SIGNIFICANCE_LEVEL else "NO",
        "α": SIGNIFICANCE_LEVEL,
    }])
    print(friedman_df.to_string(index=False))

    if p < SIGNIFICANCE_LEVEL:
        print(f"\n  → Reject H₀: at least one optimizer differs significantly (p < {SIGNIFICANCE_LEVEL})")
    else:
        print(f"\n  → Fail to reject H₀: no significant difference (p ≥ {SIGNIFICANCE_LEVEL})")

    # ── Kendall's W (effect size for Friedman) ────────────────
    print_section("KENDALL'S W (effect size)")
    w, w_interp = compute_kendalls_w(acc_matrix)
    kendall_df = pd.DataFrame([{
        "Measure": "Kendall's W",
        "Value": w,
        "Interpretation": w_interp,
    }])
    print(kendall_df.to_string(index=False))
    print(f"\n  Seeds {'agree' if w > 0.3 else 'disagree'} on optimizer ranking across runs")

    # ── Post-hoc Wilcoxon tests ───────────────────────────────
    print_section("POST-HOC: PAIRWISE WILCOXON SIGNED-RANK TESTS")
    print(f"  Correction: Holm-Bonferroni step-down (α = {SIGNIFICANCE_LEVEL})")
    posthoc_df = run_posthoc_wilcoxon(acc_matrix)

    # Clean display with formatted p-values
    display_posthoc = posthoc_df.copy()
    display_posthoc["significant"] = display_posthoc["significant"].map({True: "  YES  ", False: "  no   "})
    for col in ["p_raw", "p_corrected"]:
        if col in display_posthoc.columns:
            display_posthoc[col] = display_posthoc[col].map(lambda x: f"{x:.2e}")
    print(display_posthoc.to_string(index=False))

    sig_pairs = posthoc_df[posthoc_df["significant"]]
    if len(sig_pairs) > 0:
        print(f"\n  Significant pairs ({len(sig_pairs)}/{len(posthoc_df)}):")
        for _, row in sig_pairs.iterrows():
            print(f"    {row['opt_A']:>10s} vs {row['opt_B']:<10s}  p_corrected = {row['p_corrected']:.2e}")
    else:
        print("\n  No significant pairwise differences found.")

    # ── Generate plots ────────────────────────────────────────
    print_section("GENERATING PLOTS")
    plot_accuracy_boxplot(acc_matrix)
    plot_mean_rank_bar(acc_matrix, optimizer_order=OPTIMIZERS)
    plot_bo_convergence(trials_df)
    plot_lr_vs_accuracy(trials_df)
    plot_lr_vs_loss(trials_df)
    plot_training_curves(trials_df)
    plot_wilcoxon_heatmap(posthoc_df, list(acc_matrix.columns))
    plot_ei_over_iterations(trials_df)
    plot_early_stopping_frequency(trials_df)
    plot_fold_variance_boxplot(trials_df)
    plot_best_lr_distribution(summary_df)
    plot_val_accuracy_curves(trials_df)

    print("\nDone!")