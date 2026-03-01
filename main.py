from src.model import CNN

from torchvision import datasets, transforms
from tqdm import tqdm
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import csv
import json
import os


# ============================================================
#                         CONFIGURATION
# ============================================================

N_FOLDS = 3               # K-fold cross-validation folds (more = more robust but slower)
EPOCHS_PER_TRIAL = 7      # Training epochs per trial (more = better accuracy but slower)
EARLY_STOPPING_PATIENCE = 3  # Stop training if validation accuracy falls for this many consecutive epochs
INITIAL_POINTS = 10        # Specify points for bayesian optimization
BO_ITERATIONS = 5         # Number of iterations for bayesian optimization
BO_CANDIDATES = 100       # Set number of candidates for hyperparameter optimization

# Data settings
USE_FULL_DATASET = False   # Set to False to use subset for quick testing
SUBSET_SIZE = 6000       # Number of samples to use if USE_FULL_DATASET=False
BATCH_SIZE = 512           # Batch size for training (higher = faster but needs more memory)

# Fixed architecture (to isolate learning rate effects)
NUM_FILTERS = 32
KERNEL_SIZE = 5
NUM_UNITS = 32

# Learning rate search space
LEARNING_RATE_MIN = 0.0001
LEARNING_RATE_MAX = 0.05

# Manual seeds for reproducibility — each seed produces a fully independent run
MANUAL_SEEDS = [1, 4, 9, 12, 13, 14, 16, 17, 19, 21, 31, 35, 37, 38, 42, 45, 48, 51, 52, 55, 58, 60, 64, 66, 69, 80, 123, 256, 2048, 3651] # list of 30 random seeds for independent runs; can be extended as needed

# Optimizers to compare
OPTIMIZERS = ["SGD", "Adam", "RMSProp", "AdaGrad", "Adadelta", "AdamW", "Adamax", "ASGD", "NAdam", "RAdam", "Rprop"]

# ============================================================
#                         Functions
# ============================================================

def load_dataset():
    """
    Load the FashionMNIST dataset and return a training dataset and a testing dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def create_optimizer(name, model_params, lr):
    """
    Factory function to create an optimizer by name.
    """
    name_upper = name.upper()
    if name_upper == "SGD":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    elif name_upper == "ADAM":
        return optim.Adam(model_params, lr=lr)
    elif name_upper == "RMSPROP":
        return optim.RMSprop(model_params, lr=lr)
    elif name_upper == "ADAGRAD":
        return optim.Adagrad(model_params, lr=lr)
    elif name_upper == "ADADELTA":
        return optim.Adadelta(model_params, lr=lr)
    elif name_upper == "ADAMW":
        return optim.AdamW(model_params, lr=lr)
    elif name_upper == "ADAMAX":
        return optim.Adamax(model_params, lr=lr)
    elif name_upper == "ASGD":
        return optim.ASGD(model_params, lr=lr)
    elif name_upper == "NADAM":
        return optim.NAdam(model_params, lr=lr)
    elif name_upper == "RADAM":
        return optim.RAdam(model_params, lr=lr)
    elif name_upper == "RPROP":
        return optim.Rprop(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_model(model, train_loader, criterion, optimizer, device, epochs=EPOCHS_PER_TRIAL,
                val_loader=None, patience=EARLY_STOPPING_PATIENCE):
    """
    The training loop for the model. Returns:
      - model: trained model with best weights restored (if val_loader given)
      - epoch_losses: avg training loss per epoch (length = actual epochs run)
      - epoch_val_accs: validation accuracy after each epoch (length = actual epochs run)
      - best_epoch: 0-based index of the epoch with the highest validation accuracy
      - early_stopped: True if training was halted by early stopping
    If val_loader is provided, early stopping halts training when validation accuracy
    has not improved for `patience` consecutive epochs.
    """
    model.train()
    epoch_losses = []
    epoch_val_accs = []

    best_val_acc = -np.inf
    best_model_state = None
    best_epoch = 0
    consecutive_decreases = 0
    early_stopped = False

    for epoch_idx in tqdm(range(epochs)):
        running_loss = 0.0
        num_batches = 0
        for inputs, labels in train_loader:
            inputs_d = inputs.to(device)
            labels_d = labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs_d)

            loss = criterion(outputs, labels_d)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_losses.append(running_loss / num_batches)

        # Early stopping: evaluate on validation set after each epoch.
        if val_loader is not None:
            val_acc = evaluate_model(model, val_loader, device)
            model.train()  # Switch back to training mode after evaluation.
            epoch_val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch_idx
                consecutive_decreases = 0
            else:
                consecutive_decreases += 1
                if consecutive_decreases >= patience:
                    early_stopped = True
                    break

    # Restore the best weights seen during training.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    return model, epoch_losses, epoch_val_accs, best_epoch, early_stopped


def evaluate_model(model, test_loader, device):
    """
    Evaluation loop for the model. Returns the accuracy.
    """
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_d = inputs.to(device)
            labels_d = labels.to(device)

            outputs = model(inputs_d)

            _, predicted = torch.max(outputs.data, 1)

            total += labels_d.size(0)
            correct += (predicted == labels_d).sum().item()
    
    return correct / total


def cross_validate(train_dataset,
                   num_filters,
                   kernel_size,
                   num_units,
                   learning_rate,
                   device,
                   optimizer_name="SGD",
                   n_folds=N_FOLDS,
                   seed=42):
    """
    Computes the mean accuracy of the model by using k-fold cross validation.
    It is possible to not use the full dataset here. By default uses N_FOLDS as
    n_splits in the kFold initialization. Returns the mean accuracy and convergence data.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    all_fold_losses = []
    all_fold_val_accs = []
    fold_best_epochs = []
    fold_early_stopped = []

    # Allows us to not use the full dataset.
    if not USE_FULL_DATASET:
        indices = list(range(min(SUBSET_SIZE, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
    
    # Train the model with the parameters at the moment.
    for train_idx, val_idx in kfold.split(range(len(train_dataset))):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

        model = CNN(int(num_filters), int(kernel_size), int(num_units)).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(optimizer_name, model.parameters(), float(learning_rate))

        # Training loop and accuracy calculation.
        model, epoch_losses, epoch_val_accs, best_epoch, early_stopped = train_model(
            model, train_loader, criterion, optimizer, device, val_loader=val_loader
        )
        accuracy = evaluate_model(model, val_loader, device)
        scores.append(accuracy)
        all_fold_losses.append(epoch_losses)
        all_fold_val_accs.append(epoch_val_accs)
        fold_best_epochs.append(best_epoch)
        fold_early_stopped.append(early_stopped)
    
    # NaN-safe averaging over potentially ragged per-fold lists: early stopping can cause
    # different folds to run for different numbers of epochs.
    max_epochs = max(len(fl) for fl in all_fold_losses)
    padded_losses = [fl + [np.nan] * (max_epochs - len(fl)) for fl in all_fold_losses]
    avg_convergence = np.nanmean(padded_losses, axis=0).tolist()

    padded_val_accs = [va + [np.nan] * (max_epochs - len(va)) for va in all_fold_val_accs]
    avg_val_accs = np.nanmean(padded_val_accs, axis=0).tolist()

    return (np.mean(scores), scores, avg_convergence, all_fold_losses,
            avg_val_accs, all_fold_val_accs, fold_best_epochs, fold_early_stopped)


def objective(lr, train_dataset, device, optimizer_name="SGD", seed=42):
    """
    Objective function as a wrapper for cross validation with fixed architecture.
    Only learning rate is optimized.
    """
    (accuracy, fold_scores, convergence, fold_losses,
     avg_val_accs, fold_val_accs, fold_best_epochs, fold_early_stopped) = cross_validate(
        train_dataset,
        NUM_FILTERS,
        KERNEL_SIZE,
        NUM_UNITS,
        float(lr),
        device,
        optimizer_name=optimizer_name,
        seed=seed
    )

    return (accuracy, fold_scores, convergence, fold_losses,
            avg_val_accs, fold_val_accs, fold_best_epochs, fold_early_stopped)


def k_SE(Xi, Xj, l = 1.0, sigma = 1.0):
    """
    The gaussian process, squared exponential kernel
    """
    squared_distance = np.sum(Xi**2, axis=1).reshape(-1, 1) + np.sum(Xj**2, axis=1) - 2 * np.dot(Xi, Xj.T)

    return sigma**2 * np.exp((-1) * (squared_distance / (2 * l**2)))


def gp_posterior_predict(X_train, y_train, X_test, sigma = 1e-3):
    """
    Prediction of the posterior of standardized data, returns mu and cov.
    """
    K_with_noise = k_SE(X_train, X_train) + sigma**2 * np.eye(len(X_train))
    K_star = k_SE(X_train, X_test)
    K_starstar = k_SE(X_test, X_test)

    K_inv = np.linalg.inv(K_with_noise)

    # Matrix multiplication
    mu = K_star.T @ K_inv @ y_train
    cov = K_starstar - K_star.T @ K_inv @ K_star

    # Standard deviation
    cov = np.sqrt(np.diag(cov))

    return mu, cov #mu ko


def expected_improvement(mu, sigma, best):
    """
    Expected improvement calculated with Z as the ratio between the difference
    of mu and the best y seen and sigma. Then the result for using the formula
    for expected improvement is returned.
    """
    # Make sure sigma is not 0 and still of a reasonable value.
    sigma = np.maximum(sigma, 1e-9)

    Z = (mu - best) / sigma

    return (mu - best) * norm.cdf(Z) + sigma * norm.pdf(Z)


def normalize(X, bounds):
    """
    Normalizes the data with respect to the paramter bounds.
    """
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def bayesian_optimization(dataset, bounds, device, optimizer_name="SGD", seed=42):
    """
    Bayesian optimization (BO) for learning rate only with fixed architecture.
    Uses 1D Gaussian Process to find optimal learning rate for each optimizer.
    Returns detailed trial records.
    """
    trial_records = []  # Detailed per-trial data for CSV
    X = []
    y = []
    best_so_far = -np.inf

    # Make initial points for X and y with random learning rates.
    for i in tqdm(range(INITIAL_POINTS)):
        lr = np.random.uniform(LEARNING_RATE_MIN, LEARNING_RATE_MAX)

        (score, fold_scores, convergence, fold_losses,
         avg_val_accs, fold_val_accs, fold_best_epochs, fold_early_stopped) = objective(
            lr, dataset, device, optimizer_name, seed=seed
        )

        X.append(lr)
        y.append(score)
        best_so_far = max(best_so_far, score)

        trial_records.append({
            "iteration": i,
            "trial_type": "initial",
            "learning_rate": lr,
            "mean_accuracy": score,
            "fold_accuracies": fold_scores,
            "avg_epoch_losses": convergence,
            "fold_epoch_losses": fold_losses,
            "avg_epoch_val_accs": avg_val_accs,
            "fold_epoch_val_accs": fold_val_accs,
            "fold_best_epochs": fold_best_epochs,
            "fold_early_stopped": fold_early_stopped,
            "best_accuracy_so_far": best_so_far,
            "ei_value": None,
        })
    
    # Convert to numpy arrays for GP calculations.
    X = np.array(X).reshape(-1, 1)  # Need 2D for GP
    y = np.array(y)

    # Bayesian optimization loop.
    for i in tqdm(range(BO_ITERATIONS)):
        # Candidates uniformly chosen at random.
        candidates = np.random.uniform(
            bounds[0],
            bounds[1],
            size=(BO_CANDIDATES, 1)
        )

        # Normalize data for easier posterior calculations.
        X_norm = (X - bounds[0]) / (bounds[1] - bounds[0])
        candidates_norm = (candidates - bounds[0]) / (bounds[1] - bounds[0])

        mu, sigma = gp_posterior_predict(X_norm, y, candidates_norm)

        # Flatten results to 1D arrays.
        mu = mu.ravel()
        sigma = sigma.ravel()

        # Find the best y so far and use it with mu and sigma to calculate EI.
        best = np.max(y)
        ei = expected_improvement(mu, sigma, best)

        # next_x is the candidate at the argmax for EI.
        best_ei_idx = np.argmax(ei)
        next_lr = candidates[best_ei_idx, 0]
        best_ei_value = float(ei[best_ei_idx])

        # next_y is found by putting next_lr into the objective function.
        (next_y, fold_scores, next_convergence, fold_losses,
         avg_val_accs, fold_val_accs, fold_best_epochs, fold_early_stopped) = objective(
            next_lr, dataset, device, optimizer_name, seed=seed
        )

        # Stack and append results.
        X = np.vstack([X, [[next_lr]]])
        y = np.append(y, next_y)
        best_so_far = max(best_so_far, next_y)

        trial_records.append({
            "iteration": INITIAL_POINTS + i,
            "trial_type": "bo",
            "learning_rate": next_lr,
            "mean_accuracy": next_y,
            "fold_accuracies": fold_scores,
            "avg_epoch_losses": next_convergence,
            "fold_epoch_losses": fold_losses,
            "avg_epoch_val_accs": avg_val_accs,
            "fold_epoch_val_accs": fold_val_accs,
            "fold_best_epochs": fold_best_epochs,
            "fold_early_stopped": fold_early_stopped,
            "best_accuracy_so_far": best_so_far,
            "ei_value": best_ei_value,
        })

    return trial_records


if __name__ == "__main__":
    # Check if cuda is available. If so, use cuda. Otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    train_dataset, _ = load_dataset()

    # Define learning rate bounds (1D optimization).
    bounds = np.array([LEARNING_RATE_MIN, LEARNING_RATE_MAX])

    os.makedirs("results", exist_ok=True)

    print(f"\nFixed Architecture: filters={NUM_FILTERS}, kernel={KERNEL_SIZE}, units={NUM_UNITS}")
    print(f"Optimizing learning rate only: [{LEARNING_RATE_MIN}, {LEARNING_RATE_MAX}]\n")

    # --- Build CSV column headers dynamically based on config ---
    fold_acc_cols = [f"fold_{i+1}_accuracy" for i in range(N_FOLDS)]
    # Per-fold training metadata (epochs actually run, whether early stopping fired, best epoch).
    fold_meta_cols = (
        [f"fold_{i+1}_epochs_run"     for i in range(N_FOLDS)] +
        [f"fold_{i+1}_early_stopped"  for i in range(N_FOLDS)] +
        [f"fold_{i+1}_best_epoch"     for i in range(N_FOLDS)]
    )
    # Average (across folds) per-epoch training loss and validation accuracy.
    epoch_avg_loss_cols    = [f"epoch_{i+1}_avg_loss"    for i in range(EPOCHS_PER_TRIAL)]
    epoch_avg_val_acc_cols = [f"epoch_{i+1}_avg_val_acc" for i in range(EPOCHS_PER_TRIAL)]
    # Per-fold per-epoch training loss and validation accuracy.
    fold_epoch_loss_cols = [
        f"fold_{f+1}_epoch_{e+1}_loss"
        for f in range(N_FOLDS)
        for e in range(EPOCHS_PER_TRIAL)
    ]
    fold_epoch_val_acc_cols = [
        f"fold_{f+1}_epoch_{e+1}_val_acc"
        for f in range(N_FOLDS)
        for e in range(EPOCHS_PER_TRIAL)
    ]

    trials_header = [
        "seed", "optimizer", "iteration", "trial_type", "learning_rate",
        "mean_accuracy", "accuracy_std",
    ] + fold_acc_cols + [
        "best_accuracy_so_far", "ei_value",
    ] + fold_meta_cols + epoch_avg_loss_cols + epoch_avg_val_acc_cols + fold_epoch_loss_cols + fold_epoch_val_acc_cols

    # --- CSV formatting helpers ---
    def fmt(v):
        """Format a numeric value as a 6-decimal string; empty string for None/NaN."""
        if v is None:
            return ""
        try:
            if np.isnan(float(v)):
                return ""
        except (TypeError, ValueError):
            return str(v)
        return f"{v:.6f}"

    def pad_to(lst, length, fill=None):
        """Pad lst with fill values up to length (no-op if already long enough)."""
        return list(lst) + [fill] * max(0, length - len(lst))

    # Open CSV in write mode (overwrites on re-run).
    trials_path = "results/bo_trials.csv"
    summary_path = "results/bo_summary.csv"

    summary_rows = []

    # Build configuration comments to write at the top of CSVs
    config_comments = [
        ["# Configuration"],
        ["# N_FOLDS", N_FOLDS],    
        ["# EPOCHS_PER_TRIAL", EPOCHS_PER_TRIAL],
        ["# EARLY_STOPPING_PATIENCE", EARLY_STOPPING_PATIENCE],
        ["# INITIAL_POINTS", INITIAL_POINTS],
        ["# BO_ITERATIONS", BO_ITERATIONS],
        ["# BO_CANDIDATES", BO_CANDIDATES],
        ["# BATCH_SIZE", BATCH_SIZE],
        ["# NUM_FILTERS", NUM_FILTERS],
        ["# KERNEL_SIZE", KERNEL_SIZE],
        ["# NUM_UNITS", NUM_UNITS],
        ["# LEARNING_RATE_MIN", LEARNING_RATE_MIN],
        ["# LEARNING_RATE_MAX", LEARNING_RATE_MAX],
        ["# MANUAL_SEEDS", str(MANUAL_SEEDS)],
    ]

    with open(trials_path, "w", newline="") as trials_file:
        writer = csv.writer(trials_file)
        for comment_row in config_comments:
            writer.writerow(comment_row)
        writer.writerow(trials_header)

        for seed in MANUAL_SEEDS:
          for opt_name in OPTIMIZERS:
            print(f"\n{'='*60}")
            print(f"  Optimizing Learning Rate for {opt_name} (seed={seed})")
            print(f"{'='*60}\n")

            # Reset seeds for fair comparison across optimizers.
            torch.manual_seed(seed)
            np.random.seed(seed)

            trial_records = bayesian_optimization(
                train_dataset, bounds, device, optimizer_name=opt_name, seed=seed
            )

            # Write each trial to CSV.
            for rec in trial_records:
                lr                  = rec["learning_rate"]
                fold_accs           = rec["fold_accuracies"]
                avg_losses          = rec["avg_epoch_losses"]
                fold_losses         = rec["fold_epoch_losses"]       # list[list[float]]
                avg_val_accs_rec    = rec["avg_epoch_val_accs"]      # list[float], len <= EPOCHS_PER_TRIAL
                fold_val_accs       = rec["fold_epoch_val_accs"]     # list[list[float]]
                fold_best_epochs_rec  = rec["fold_best_epochs"]      # list[int]
                fold_early_stopped_rec = rec["fold_early_stopped"]   # list[bool]

                # Per-fold epoch training losses, padded to EPOCHS_PER_TRIAL.
                flat_fold_losses = []
                for fl in fold_losses:
                    flat_fold_losses.extend([fmt(v) for v in pad_to(fl, EPOCHS_PER_TRIAL)])

                # Per-fold epoch validation accuracies, padded to EPOCHS_PER_TRIAL.
                flat_fold_val_accs = []
                for fv in fold_val_accs:
                    flat_fold_val_accs.extend([fmt(v) for v in pad_to(fv, EPOCHS_PER_TRIAL)])

                row = [
                    seed,
                    opt_name,
                    rec["iteration"],
                    rec["trial_type"],
                    fmt(float(lr)),
                    fmt(rec["mean_accuracy"]),
                    fmt(np.std(fold_accs)),
                ] + [
                    fmt(a) for a in fold_accs                              # fold_N_accuracy
                ] + [
                    fmt(rec["best_accuracy_so_far"]),
                    fmt(rec["ei_value"]) if rec["ei_value"] is not None else "",
                ] + [
                    str(len(fl)) for fl in fold_losses                     # fold_N_epochs_run
                ] + [
                    str(1 if es else 0) for es in fold_early_stopped_rec  # fold_N_early_stopped
                ] + [
                    str(be) for be in fold_best_epochs_rec                # fold_N_best_epoch
                ] + [
                    fmt(v) for v in pad_to(avg_losses, EPOCHS_PER_TRIAL)  # epoch_N_avg_loss
                ] + [
                    fmt(v) for v in pad_to(avg_val_accs_rec, EPOCHS_PER_TRIAL)  # epoch_N_avg_val_acc
                ] + flat_fold_losses + flat_fold_val_accs

                writer.writerow(row)

            # Flush after each optimizer so partial results are saved.
            trials_file.flush()

            # Find the best trial for this optimizer.
            best_rec = max(trial_records, key=lambda r: r["mean_accuracy"])
            best_lr = best_rec["learning_rate"]

            summary_rows.append({
                "seed": seed,
                "optimizer": opt_name,
                "best_accuracy": best_rec["mean_accuracy"],
                "accuracy_std": float(np.std(best_rec["fold_accuracies"])),
                "best_learning_rate": float(best_lr),
                "best_final_loss": best_rec["avg_epoch_losses"][-1],
                "total_trials": len(trial_records),
            })

            print(f"\n{opt_name} (seed={seed}) Results:")
            print(f"  Best accuracy: {best_rec['mean_accuracy']:.4f} "
                  f"(±{np.std(best_rec['fold_accuracies']):.4f})")
            print(f"  Best learning rate: {float(best_lr):.6f}")

    # Write summary CSV (one row per optimizer).
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        for comment_row in config_comments:
            writer.writerow(comment_row)
        summary_writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        summary_writer.writeheader()
        summary_writer.writerows(summary_rows)

    print(f"\nResults saved to:")
    print(f"  Trials:  {trials_path}")
    print(f"  Summary: {summary_path}")

    # Print summary table.
    print(f"\n{'='*60}")
    print(f"  Summary (Fixed arch: {NUM_FILTERS} filters, {KERNEL_SIZE}x{KERNEL_SIZE} kernel, {NUM_UNITS} units)")
    print(f"{'='*60}")
    for row in summary_rows:
        print(f"  {row['optimizer']:>8s} (seed={row['seed']}): accuracy = {row['best_accuracy']:.4f} "
              f"(±{row['accuracy_std']:.4f}), lr = {row['best_learning_rate']:.6f}")
