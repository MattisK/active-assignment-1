import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
import optuna
import matplotlib.pyplot as plt

# Config
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

TRIALS = 15
EPOCHS_PER_TRIAL = 1
FINAL_EPOCHS = 10
BATCH_SIZE = 128
VAL_SIZE = 10_000

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_full = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

train_set, val_set = random_split(
    train_full,
    [len(train_full) - VAL_SIZE, VAL_SIZE],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Model (tiny CNN)
class TinyCNN(nn.Module):
    def __init__(self, channels=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 28 -> 14
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 14 -> 7
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear((channels * 2) * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)

def train_epochs(model, loader, optimizer, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

# Objective (hyperparams -> val acc)
# Tune only 3 params: lr, weight_decay, dropout
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    model = TinyCNN(channels=32, dropout=dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    train_epochs(model, train_loader, opt, EPOCHS_PER_TRIAL)
    val_acc = accuracy(model, val_loader)
    return val_acc

def run_study(name, sampler):
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=name)
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
    return study

def best_so_far(study):
    vals = [t.value for t in study.trials if t.value is not None]
    out, m = [], -1
    for v in vals:
        m = max(m, v)
        out.append(m)
    return out

def retrain_and_test(best_params):
    full_train = ConcatDataset([train_set, val_set])
    full_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True)

    model = TinyCNN(channels=32, dropout=best_params["dropout"]).to(device)
    opt = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

    train_epochs(model, full_loader, opt, FINAL_EPOCHS)
    return accuracy(model, test_loader)

if __name__ == "__main__":
    print("Device:", device)

    # "Bayesian optimization" in Optuna = TPE (SMBO)
    tpe = run_study("TPE", optuna.samplers.TPESampler(seed=SEED))
    rnd = run_study("Random", optuna.samplers.RandomSampler(seed=SEED))

    print("\nBest val (TPE):", tpe.best_value, tpe.best_params)
    print("Best val (Random):", rnd.best_value, rnd.best_params)

    tpe_test = retrain_and_test(tpe.best_params)
    rnd_test = retrain_and_test(rnd.best_params)

    print("\nFinal test acc (retrain best):")
    print("TPE:", tpe_test)
    print("Random:", rnd_test)

    plt.figure()
    plt.plot(best_so_far(tpe), label="TPE (BO/SMBO)")
    plt.plot(best_so_far(rnd), label="Random")
    plt.xlabel("Trial")
    plt.ylabel("Best validation accuracy so far")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    print("\nSaved: convergence.png")