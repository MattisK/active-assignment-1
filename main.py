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


# ============================================================
#                         CONFIGURATION
# ============================================================

N_FOLDS = 5               # K-fold cross-validation folds (more = more robust but slower)
EPOCHS_PER_TRIAL = 1      # Training epochs per trial (more = better accuracy but slower)
INITIAL_POINTS = 1        # Specify points for bayesian optimization
BO_ITERATIONS = 1         # Number of iterations for bayesian optimization
BO_CANDIDATES = 200       # Set number of candidates for hyperparameter optimization

# Data settings
USE_FULL_DATASET = True   # Set to False to use subset for quick testing
SUBSET_SIZE = 10000       # Number of samples to use if USE_FULL_DATASET=False
BATCH_SIZE = 64           # Batch size for training (higher = faster but needs more memory)

# Hyperparameter search space (ranges to explore)
LEARNING_RATE_MIN = 0.001
LEARNING_RATE_MAX = 0.1
NUM_FILTERS_MIN = 16      # Minimum convolutional filters
NUM_FILTERS_MAX = 64      # Maximum convolutional filters
KERNEL_SIZE_MIN = 3       # Minimum kernel size (must be odd)
KERNEL_SIZE_MAX = 7       # Maximum kernel size (must be odd)
NUM_UNITS_MIN = 50        # Minimum units in dense layer
NUM_UNITS_MAX = 200       # Maximum units in dense layer

# Manual seed for reproducibility
MANUAL_SEED = 0

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


def train_model(model, train_loader, criterion, optimizer, device, epochs=EPOCHS_PER_TRIAL):
    """
    The training loop for the model.
    """
    model.train()

    for _ in tqdm(range(epochs)):
        for inputs, labels in train_loader:
            inputs_d = inputs.to(device)
            labels_d = labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs_d)

            loss = criterion(outputs, labels_d)
            loss.backward()
            optimizer.step()
            
    return model


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
                   n_folds=N_FOLDS):
    """
    Computes the mean accuracy of the model by using k-fold cross validation.
    It is possible to not use the full dataset here. By default uses N_FOLDS as
    n_splits in the kFold initialization. Returns the mean accuracy.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=MANUAL_SEED)
    scores = []

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
        optimizer = optim.SGD(model.parameters(), lr=float(learning_rate), momentum=0.9)

        # Training loop and accuracy calculation.
        model = train_model(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate_model(model, val_loader, device)
        scores.append(accuracy)
    
    return np.mean(scores)


def objective(params, train_dataset, device):
    """
    Objective function as a wrapper for the cross validation, given the parameters we want to optimize.
    """
    num_filters, kernel_size, num_units, lr = params

    # Kernel size must be odd.
    if int(kernel_size) % 2 == 0:
        kernel_size += 1

    accuracy = cross_validate(
        train_dataset,
        int(num_filters),
        int(kernel_size),
        int(num_units),
        float(lr),
        device
    )

    return accuracy


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

    return mu, cov


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
    return (X - bounds[:, 0] / bounds[:, 1] - bounds[:, 0])


def bayesian_optimization(dataset, bounds, device):
    """
    Bayesian optimization (BO) on a given dataset with specified bounds.
    Stores the best accuracy and corresponding parameters for each BO iteration.
    These are computed by normalizing the data, then compute the posterior and finding
    the expected improvement (EI). The EI is used to find the max argument for the
    candidates, this value is then put through the objective function to find the next y.
    Returns the accuracy and parameters in a list by BO iteration.
    """
    best_accuracy = []
    best_params = []

    X = []
    y = []

    # Make intial points for X and y with random parameters.
    for _ in tqdm(range(INITIAL_POINTS)):
        params = np.array([
            np.random.randint(NUM_FILTERS_MIN, NUM_FILTERS_MAX),
            np.random.randint(KERNEL_SIZE_MIN, KERNEL_SIZE_MAX),
            np.random.randint(NUM_UNITS_MIN, NUM_UNITS_MAX),
            np.random.uniform(LEARNING_RATE_MIN, LEARNING_RATE_MAX)            
        ])

        score = objective(params, dataset, device)

        X.append(params)
        y.append(score)
    
    # Convert to a numpy array for later calculations.
    X = np.array(X)
    y = np.array(y)

    # Bayesian optimization loop.
    for _ in tqdm(range(BO_ITERATIONS)):
        # Candidates uniformly chosen at random.
        candidates = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(BO_CANDIDATES, 4)
        )

        # Normalize data for easier posterior calculations.
        X_norm = normalize(X, bounds)
        candidates_norm = normalize(candidates, bounds)

        mu, sigma = gp_posterior_predict(X_norm, y, candidates_norm)

        # Flatten results to 1D arrays.
        mu = mu.ravel()
        sigma = sigma.ravel()

        # Find the best y so far and use it with mu and sigma to calculate the posterior.
        best = np.max(y)
        ei = expected_improvement(mu, sigma, best)

        # next_x is the candidate at the argmax for EI.
        next_x = candidates[np.argmax(ei)]

        # next_y is found by putting next_x into the objective function.
        next_y = objective(next_x, dataset, device)

        # Stack and append results.
        X = np.vstack([X, next_x])
        y = np.append(y, next_y)

        # Save results.
        best_accuracy.append(np.max(y))
        best_params.append(next_x)

    return best_accuracy, best_params


if __name__ == "__main__":
    # Set the manual seed in Pytorch and Numpy.
    torch.manual_seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)

    # Check if cuda is available. If so, use cuda. Otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    train_dataset, _ = load_dataset()

    # Define boundaries based on the config.
    bounds = np.array([
        [NUM_FILTERS_MIN, NUM_FILTERS_MAX],
        [KERNEL_SIZE_MIN, KERNEL_SIZE_MAX],
        [NUM_UNITS_MIN, NUM_UNITS_MAX],
        [LEARNING_RATE_MIN, LEARNING_RATE_MAX]
    ])

    # Bayesian optimization.
    best_accuracy, best_params = bayesian_optimization(train_dataset, bounds, device)

    print(best_accuracy, best_params)
