# PyTorch CNN model for Fashion MNIST with Bayesian Optimization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import optuna

# ============================================================
# CONFIGURATION - Adjust these parameters for tuning
# ============================================================

# Optimization settings
NUM_TRIALS = 50           # Number of hyperparameter combinations to try (more = better but slower)
N_FOLDS = 5               # K-fold cross-validation folds (more = more robust but slower)
EPOCHS_PER_TRIAL = 10     # Training epochs per trial (more = better accuracy but slower)

# Data settings
USE_FULL_DATASET = True   # Set to False to use subset for quick testing
SUBSET_SIZE = 10000       # Number of samples to use if USE_FULL_DATASET=False
BATCH_SIZE = 64           # Batch size for training (higher = faster but needs more memory)

# Hyperparameter search space (ranges to explore)
LEARNING_RATE_MIN = 0.001
LEARNING_RATE_MAX = 0.1
NUM_FILTERS_MIN = 16      # Minimum convolutional filters
NUM_FILTERS_MAX = 64      # Maximum convolutional filters
FILTERS_STEP = 8
KERNEL_SIZE_MIN = 3       # Minimum kernel size (must be odd)
KERNEL_SIZE_MAX = 7       # Maximum kernel size (must be odd)
KERNEL_SIZE_STEP = 2
NUM_UNITS_MIN = 50        # Minimum units in dense layer
NUM_UNITS_MAX = 200       # Maximum units in dense layer
UNITS_STEP = 25

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device configuration
device = torch.device('cpu')  # Intel Mac compatibility

# Load Fashion MNIST dataset
def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_filters, kernel_size, num_units):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Calculate the flattened size after conv and pooling
        conv_output_size = num_filters * 14 * 14
        
        self.fc1 = nn.Linear(conv_output_size, num_units)
        self.fc2 = nn.Linear(num_units, 10)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS_PER_TRIAL):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

# Cross-validation with hyperparameters
def cross_validate(train_dataset, num_filters, kernel_size, num_units, learning_rate, n_folds=N_FOLDS):
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    
    # Use subset for faster training if configured
    from torch.utils.data import Subset
    if not USE_FULL_DATASET:
        indices = list(range(min(SUBSET_SIZE, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
    
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_dataset)))):
        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
        
        # Create model
        model = SimpleCNN(num_filters, kernel_size, num_units).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        # Train and evaluate
        model = train_model(model, train_loader, criterion, optimizer)
        accuracy = evaluate_model(model, val_loader)
        scores.append(accuracy)
    
    mean_accuracy = np.mean(scores)
    return mean_accuracy

# Global variable to store dataset
train_dataset = None

# Objective function for Optuna
def objective(trial):
    global train_dataset
    
    # Suggest hyperparameters from configured search space
    learning_rate = trial.suggest_float('learning_rate', LEARNING_RATE_MIN, LEARNING_RATE_MAX, log=True)
    num_filters = trial.suggest_int('num_filters', NUM_FILTERS_MIN, NUM_FILTERS_MAX, step=FILTERS_STEP)
    kernel_size = trial.suggest_int('kernel_size', KERNEL_SIZE_MIN, KERNEL_SIZE_MAX, step=KERNEL_SIZE_STEP)
    num_units = trial.suggest_int('num_units', NUM_UNITS_MIN, NUM_UNITS_MAX, step=UNITS_STEP)
    
    print(f"\n[Trial {trial.number}] filters={num_filters}, kernel={kernel_size}, "
          f"units={num_units}, lr={learning_rate:.4f}")
    
    accuracy = cross_validate(train_dataset, num_filters, kernel_size, num_units, learning_rate)
    
    print(f"  → CV Accuracy: {accuracy*100:.2f}%")
    
    return accuracy

# Main execution
if __name__ == '__main__':
    print("Starting Bayesian Optimization for Fashion MNIST CNN")
    print("=" * 60)
    
    # Load dataset once
    train_dataset, _ = load_dataset()
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Trials: {NUM_TRIALS}")
    print(f"  Cross-validation folds: {N_FOLDS}")
    print(f"  Epochs per trial: {EPOCHS_PER_TRIAL}")
    print(f"  Dataset: {'Full (60,000)' if USE_FULL_DATASET else f'Subset ({SUBSET_SIZE})'}")
    print(f"  Batch size: {BATCH_SIZE}")
    print()
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest CV Accuracy: {study.best_value*100:.2f}%")
    
    # Extract scores for plotting
    trials_df = study.trials_dataframe()
    best_scores = trials_df['value'].cummax()
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    sns.set_palette("husl")
    sns.despine()
    plt.plot(range(len(best_scores)), best_scores*100, linewidth=2, marker='o', markersize=4)
    plt.title("Bayesian Optimization Convergence", fontsize=16, fontweight='bold')
    plt.xlabel('Trial', fontsize=12)
    plt.ylabel('Best CV Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimization_convergence.png', dpi=150)
    print("\n✓ Saved convergence plot to 'optimization_convergence.png'")
    
    # Plot learning rate distribution
    plt.figure(figsize=(10, 6))
    sns.set_palette("husl")
    sns.despine()
    learning_rates = trials_df['params_learning_rate']
    sns.histplot(learning_rates, bins=15, kde=True)
    plt.title("Learning Rate Exploration", fontsize=16, fontweight='bold')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('learning_rate_distribution.png', dpi=150)
    print("✓ Saved learning rate plot to 'learning_rate_distribution.png'")
    
    # Plot units distribution
    plt.figure(figsize=(10, 6))
    sns.set_palette("husl")
    sns.despine()
    units = trials_df['params_num_units']
    sns.histplot(units, bins=10, kde=True)
    plt.title("Number of Units Exploration", fontsize=16, fontweight='bold')
    plt.xlabel('Number of Units', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('units_distribution.png', dpi=150)
    print("✓ Saved units plot to 'units_distribution.png'")
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(10, 6))
        sns.set_palette("husl")
        sns.despine()
        params = list(importance.keys())
        values = list(importance.values())
        plt.barh(params, values)
        plt.title("Hyperparameter Importance", fontsize=16, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.tight_layout()
        plt.savefig('parameter_importance.png', dpi=150)
        print("✓ Saved parameter importance plot to 'parameter_importance.png'")
    except:
        pass
    
    print("\nDone!")

