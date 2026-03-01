from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt


def _as_history_dict(history):
	if hasattr(history, "history") and isinstance(history.history, Mapping):
		return history.history
	if isinstance(history, Mapping):
		return history
	if isinstance(history, Sequence) and len(history) == 2:
		train_values, val_values = history
		return {
			"train_accuracy": train_values,
			"val_accuracy": val_values,
		}
	raise TypeError(
		"history must be dict-like, an object with a .history dict attribute, "
		"or a (train_values, val_values) sequence"
	)


def _find_first_available_key(history_dict, keys):
	for key in keys:
		if key in history_dict:
			return key
	return None


def _is_number_sequence(values):
	return isinstance(values, Sequence) and not isinstance(values, (str, bytes))


def plot_accuracy_history(
	history,
	train_keys=("train_accuracy", "train_acc", "accuracy", "acc"),
	val_keys=("val_accuracy", "val_acc", "validation_accuracy"),
	title="Training vs Validation Accuracy",
	save_path=None,
	show=True,
	ax=None,
):
	"""Plot training and validation accuracy curves from a model history object.

	Parameters
	----------
	history:
		Either a dict-like history or an object exposing `.history` (e.g., Keras History).
	train_keys:
		Candidate keys for training accuracy series.
	val_keys:
		Candidate keys for validation accuracy series.
	title:
		Plot title.
	save_path:
		Optional file path to save the figure.
	show:
		If True, displays the figure.
	ax:
		Optional matplotlib axis to draw on.
	"""
	history_dict = _as_history_dict(history)

	train_key = _find_first_available_key(history_dict, train_keys)
	val_key = _find_first_available_key(history_dict, val_keys)

	if train_key is None and val_key is None:
		available = ", ".join(history_dict.keys())
		raise ValueError(
			"No accuracy keys found in history. "
			f"Looked for train keys {train_keys} and val keys {val_keys}. "
			f"Available keys: {available}"
		)

	if ax is None:
		_, ax = plt.subplots(figsize=(9, 5))

	if train_key is not None:
		train_values = history_dict[train_key]
		if not _is_number_sequence(train_values):
			raise TypeError(f"Expected sequence for '{train_key}', got {type(train_values)}")
		ax.plot(train_values, label="Train Accuracy", linewidth=2)

	if val_key is not None:
		val_values = history_dict[val_key]
		if not _is_number_sequence(val_values):
			raise TypeError(f"Expected sequence for '{val_key}', got {type(val_values)}")
		ax.plot(val_values, label="Validation Accuracy", linewidth=2)

	ax.set_title(title)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Accuracy")
	ax.grid(alpha=0.3)
	ax.legend()

	if save_path:
		ax.figure.tight_layout()
		ax.figure.savefig(save_path, dpi=150)

	if show:
		plt.show()

	return ax


def plot_learning_rate_history(
	optimizer_lr_history,
	title="Learning Rate History by Optimizer",
	save_path=None,
	show=True,
	ax=None,
):
	"""Plot learning-rate schedules for one or multiple optimizers.

	Parameters
	----------
	optimizer_lr_history:
		Mapping with optimizer names as keys and lr sequences as values.
		Example: {"Adam": [1e-3, 8e-4], "SGD": [1e-2, 7e-3]}
	title:
		Plot title.
	save_path:
		Optional file path to save the figure.
	show:
		If True, displays the figure.
	ax:
		Optional matplotlib axis to draw on.
	"""
	if isinstance(optimizer_lr_history, Sequence) and not isinstance(
		optimizer_lr_history, (str, bytes)
	):
		normalized = {}
		for run in optimizer_lr_history:
			if not isinstance(run, Mapping) or "optimizer" not in run or "lr" not in run:
				raise TypeError(
					"When using sequence input, each item must look like "
					"{'optimizer': <name>, 'lr': <sequence or float>}"
				)
			optimizer_name = str(run["optimizer"])
			lr_values = run["lr"]
			if _is_number_sequence(lr_values):
				normalized.setdefault(optimizer_name, []).extend(lr_values)
			else:
				normalized.setdefault(optimizer_name, []).append(lr_values)
		optimizer_lr_history = normalized

	if not isinstance(optimizer_lr_history, Mapping) or not optimizer_lr_history:
		raise TypeError(
			"optimizer_lr_history must be a non-empty mapping of "
			"{optimizer_name: lr_sequence} or a sequence of {'optimizer', 'lr'} items"
		)

	if ax is None:
		_, ax = plt.subplots(figsize=(9, 5))

	for optimizer_name, lr_values in optimizer_lr_history.items():
		if not _is_number_sequence(lr_values):
			raise TypeError(
				f"Learning-rate values for '{optimizer_name}' must be a sequence, "
				f"got {type(lr_values)}"
			)
		ax.plot(lr_values, label=str(optimizer_name), linewidth=2)

	ax.set_title(title)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Learning Rate")
	ax.grid(alpha=0.3)
	ax.legend(title="Optimizer")

	if save_path:
		ax.figure.tight_layout()
		ax.figure.savefig(save_path, dpi=150)

	if show:
		plt.show()

	return ax

