# Hyperparameter Tuning Guide

## Quick Start

### Run Full Optimization (Production Mode)
```bash
# Already configured for full run in model.py
uv run python src/model.py
```

The current settings in `model.py` are optimized for thorough search:
- **50 trials** - tests 50 different hyperparameter combinations
- **5-fold CV** - robust validation
- **10 epochs** per trial
- **Full dataset** (60,000 samples)

**Estimated time:** ~2-3 hours on Intel Mac

### Run Quick Test
Edit the top of `src/model.py` and change:
```python
NUM_TRIALS = 10           # Fewer trials
N_FOLDS = 2               # Fewer folds
EPOCHS_PER_TRIAL = 3      # Fewer epochs
USE_FULL_DATASET = False  # Use subset
```

**Estimated time:** ~5-10 minutes

## Configuration Parameters

### Located at top of `src/model.py` (lines 14-46)

#### Optimization Settings

| Parameter | Default | What it does | Recommendation |
|-----------|---------|--------------|----------------|
| `NUM_TRIALS` | 50 | Number of hyperparameter combinations to try | **Quick:** 10, **Thorough:** 50-100 |
| `N_FOLDS` | 5 | Cross-validation folds for robustness | **Min:** 2, **Good:** 5, **Best:** 10 |
| `EPOCHS_PER_TRIAL` | 10 | Training epochs per trial | **Quick:** 3-5, **Good:** 10, **Best:** 20 |

#### Data Settings

| Parameter | Default | What it does | Recommendation |
|-----------|---------|--------------|----------------|
| `USE_FULL_DATASET` | True | Use all 60k samples vs subset | **Testing:** False, **Production:** True |
| `SUBSET_SIZE` | 10000 | Samples to use if not using full | 5000-20000 for testing |
| `BATCH_SIZE` | 64 | Training batch size | 32-128 (higher = faster, needs more RAM) |

#### Hyperparameter Search Space

Define the ranges Bayesian optimization will explore:

| Parameter | Default Range | What it controls |
|-----------|---------------|------------------|
| `LEARNING_RATE_MIN/MAX` | 0.001 - 0.1 | How fast the model learns |
| `NUM_FILTERS_MIN/MAX` | 16 - 64 | Conv layer capacity (more = more features but slower) |
| `KERNEL_SIZE_MIN/MAX` | 3 - 7 | Conv filter size (larger = bigger patterns) |
| `NUM_UNITS_MIN/MAX` | 50 - 200 | Dense layer size |

## How to Adjust for Different Goals

### Maximum Accuracy (Don't care about time)
```python
NUM_TRIALS = 100
N_FOLDS = 10
EPOCHS_PER_TRIAL = 20
USE_FULL_DATASET = True
BATCH_SIZE = 32  # Smaller for better gradient estimates
```

### Quick Experimentation
```python
NUM_TRIALS = 10
N_FOLDS = 2
EPOCHS_PER_TRIAL = 3
USE_FULL_DATASET = False
SUBSET_SIZE = 5000
BATCH_SIZE = 128
```

### Balanced (Current Default)
```python
NUM_TRIALS = 50
N_FOLDS = 5
EPOCHS_PER_TRIAL = 10
USE_FULL_DATASET = True
BATCH_SIZE = 64
```

### Explore Different Model Architectures
Expand the search space:
```python
NUM_FILTERS_MIN = 8
NUM_FILTERS_MAX = 128
NUM_UNITS_MIN = 25
NUM_UNITS_MAX = 400
```

## Understanding the Output

After optimization completes, you'll get:

1. **Best hyperparameters** - The optimal combination found
2. **CV Accuracy** - Expected accuracy on new data
3. **4 visualization plots:**
   - `optimization_convergence.png` - How quickly best params were found
   - `learning_rate_distribution.png` - Learning rates tested
   - `units_distribution.png` - Dense layer sizes tested
   - `parameter_importance.png` - Which params matter most

## Training Time Estimates

**Intel Mac CPU:**
- 1 trial × 1 fold × 1 epoch × 10k samples ≈ 10-15 seconds
- Full config (50 trials × 5 folds × 10 epochs × 60k samples) ≈ 2-3 hours

**Tips to speed up:**
1. Reduce `NUM_TRIALS` first (biggest impact)
2. Then reduce `N_FOLDS`
3. Then reduce `EPOCHS_PER_TRIAL`
4. Finally, use subset if needed

## Example Workflow

### Step 1: Quick exploration (5 min)
```python
NUM_TRIALS = 10
N_FOLDS = 2
USE_FULL_DATASET = False
```

### Step 2: Refine search space
Look at plots, adjust ranges to focus on promising areas

### Step 3: Full optimization (2-3 hours)
```python
NUM_TRIALS = 50-100
N_FOLDS = 5
USE_FULL_DATASET = True
```

## Advanced: Custom Hyperparameters

To add new hyperparameters to optimize, edit the `objective()` function:

```python
def objective(trial):
    # Add new parameter
    dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Use it in cross_validate or model definition
    ...
```

Then update the model architecture to use it.
