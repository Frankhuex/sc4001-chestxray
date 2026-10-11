# SC4001 Part B: Chest X-ray Classification

This README is the shared run guide for Part B. B1 is implemented; B2–B4 will be added as their experiments are completed. Run all commands from the repository root.

## Environment and data

Use Python 3.12 (see `.python-version`). Install the project dependencies in a local environment with `uv sync`, then run scripts with `uv run python`. This creates the project-local `.venv`; `uv.lock` records the resolved dependency versions. Subsequent work uses this project environment. Alternatively, use an existing Python environment with NumPy, PyTorch and Matplotlib installed. Place `chestxray_binary_64.npz` in the project root. The latest B1 run used the project-local environment with Python 3.12.12, PyTorch 2.14.1, NumPy 2.5.4 and Matplotlib 3.11.2 on CPU; exact settings and preprocessing statistics are recorded in `outputs/b1/config.json`.

## Project layout

- `src/`: experiment scripts.
- `outputs/b1/`: B1 results, plots, checkpoints, logs and source snapshots; `outputs/` is ignored by Git.
- `report/PartB_report_HuHan.tex`: the Part B report source.

## B1: learning-rate experiments

```sh
uv run python src/train_alexnet_learning_rates.py --data chestxray_binary_64.npz --output outputs/b1 --seed 42 --threads 6
```

The supplied train/validation split is retained. Images are divided by 255 and standardized with the pixel-wise training-set mean and population standard deviation. No augmentation or class weighting is used. Label 1 is treated as pneumonia (positive), label 0 as healthy; the NPZ contains numeric labels without class-name metadata. A sigmoid probability of at least 0.5 predicts the positive class.

Architecture: Conv(1,16,5)–BN–ReLU–MaxPool(2), Conv(16,32,5)–BN–ReLU–MaxPool(2), Conv(32,64,3)–BN–ReLU–MaxPool(2), adaptive average pool to 4×4, FC(1024,64)–ReLU–Dropout(0.5), FC(64,32)–ReLU–Dropout(0.5), FC(32,1). SAME convolution padding and dropout placement follow the assignment slides. Convolutions have no bias; fully connected layers have bias. The output is a raw binary logit for BCEWithLogitsLoss; sigmoid is implicit in the loss and classification threshold.

All five runs use the same seed, initial weights, independently seeded shuffle generator, batch size 128, Adam defaults except learning rate, maximum 20 epochs, validation-loss early stopping with patience 4 and min_delta 1e-4. Early stopping counts epochs without a loss decrease exceeding min_delta. Each run's selected checkpoint minimizes validation classification error; ties use lower validation loss, then the earlier epoch. This rule is independent of early-stopping min_delta. Curves include every executed epoch and terminate at the actual stopping epoch.

Outputs: `lr_*_history.csv` and `histories.json` contain per-epoch metrics; `summary.csv` contains the selected epoch metrics; `lr_*_best.pt` stores selected weights and preprocessing; separate validation-loss and validation-accuracy figures are provided as PDF and PNG; `run.log` records progress. Sensitivity = TP/(TP+FN); specificity = TN/(TN+FP). The training mode does not read or evaluate the test set. Evaluate the validation-selected checkpoint once for B1(d):

```sh
uv run python src/train_alexnet_learning_rates.py --evaluate-only --data chestxray_binary_64.npz --output outputs/b1 --threads 6
```

This mode uses the saved training normalisation statistics and writes `test_metrics.json` and `test_metrics.csv` in `outputs/b1`.

## Report compilation

The report embeds plot coordinates and can be compiled in the built-in LaTeX editor. With a local TeX installation, compile from the report directory:

```sh
cd report
pdflatex -interaction=nonstopmode -halt-on-error PartB_report_HuHan.tex
```

B2–B4 run commands and output descriptions will be added here alongside their implementations.
