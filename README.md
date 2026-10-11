# SC4001 Part B: Chest X-ray Classification

This README is the shared run guide for Part B. B1–B3 are implemented; B4 will be added as it is completed. Run all commands from the repository root.

## Environment and data

Use Python 3.12 (see `.python-version`). Install the project dependencies in a local environment with `uv sync`, then run scripts with `uv run python`. This creates the project-local `.venv`; `uv.lock` records the resolved dependency versions. Subsequent work uses this project environment. Alternatively, use an existing Python environment with NumPy, PyTorch and Matplotlib installed. Place `chestxray_binary_64.npz` in the project root. The latest B1 run used the project-local environment with Python 3.12.12, PyTorch 2.14.1, NumPy 2.5.4 and Matplotlib 3.11.2 on CPU; exact settings and preprocessing statistics are recorded in `outputs/b1/config.json`.

## Project layout

- `src/`: experiment scripts.
- `outputs/b1/`, `outputs/b2/` and `outputs/b3/`: experiment results, plots, checkpoints, logs and source snapshots; `outputs/` is ignored by Git.
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

## B2: VGG batch-size experiments

```sh
uv run python -u src/train_vgg_batch_sizes.py --data chestxray_binary_64.npz --output outputs/b2 --learning-rate 1e-3 --seed 42 --threads 6
```

The VGG-like model uses two 3×3 convolutions at each width (8, 16, 32), with BatchNorm and ReLU after every convolution and 2×2 max pooling after each pair. Adaptive average pooling produces a 1×1 output, followed by a biased linear layer (32→1). Convolutions have SAME padding and no bias; B2 uses no dropout, following the supplementary slides.

Learning rate is fixed to B1's selected 1e-3; batch sizes 16, 32, 64, 128 and 256 are compared. Preprocessing, seed 42, Adam and stopping/checkpoint rules match B1. Initial weights and shuffle seeds are matched; changing batch size changes minibatch grouping and updates per epoch. Validation evaluation uses a fixed batch size of 128. The training command does not read the test split.

`outputs/b2` contains `batch_*_history.csv`, `histories.json`, `summary.csv`, `batch_*_best.pt`, `config.json`, and separate validation-loss/accuracy plots in PDF and PNG. To retain a visible console log, pipe the unbuffered training command to `tee outputs/b2/run.log` after creating the output directory. Evaluate the validation-selected checkpoint once:

```sh
uv run python -u src/train_vgg_batch_sizes.py --evaluate-only --data chestxray_binary_64.npz --output outputs/b2 --threads 6
```

Final results are saved as `outputs/b2/test_metrics.json` and `test_metrics.csv`.

## B3: fixed ResNet classifier

```sh
uv run python -u src/train_resnet_classifier.py --data chestxray_binary_64.npz --output outputs/b3 --learning-rate 1e-3 --batch-size 16 --seed 42 --threads 6
```

The fixed learning rate and batch size come from B1 and B2. A 3×3, 16-channel stem and max pooling precede residual blocks of widths 16, 32 and 64. The latter two blocks downsample with stride 2 and use 1×1 projection shortcuts. Every convolution has no bias and is followed by BatchNorm and ReLU; residual additions also use ReLU. Global average pooling and a biased 64→1 layer produce a binary logit. No dropout is used. Preprocessing, Adam, early stopping and checkpoint-selection rules match B1–B2.

The script reuses typed metric/CSV helpers and the checkpoint schema from the other scripts in `src/`. `outputs/b3` contains `history.csv`, `history.json`, `summary.csv`, `config.json`, `best.pt` and training/validation loss curves (`training_curves.pdf` and `.png`). Evaluate the selected checkpoint once:

```sh
uv run python -u src/train_resnet_classifier.py --evaluate-only --data chestxray_binary_64.npz --output outputs/b3 --threads 6
```

Test results are written to `outputs/b3/test_metrics.json` and `test_metrics.csv`. For live logs, append `| tee outputs/b3/run.log` to training, or `| tee outputs/b3/evaluation.log` to evaluation, after creating the output directory.

## Architecture comparison

```sh
uv run python src/compare_model_complexity.py
```

This reads the saved B1–B3 test metrics without repeating test evaluation. Forward hooks count convolution and linear MACs on one 1×64×64 dummy input; FLOPs = 2×MACs. BatchNorm, activations, pooling, bias additions and residual additions are excluded. All trainable parameters are counted. The comparison is saved to `outputs/b3/architecture_comparison.csv` and `.json`.

## Report compilation

The report embeds plot coordinates and can be compiled in the built-in LaTeX editor. With a local TeX installation, compile from the report directory:

```sh
cd report
pdflatex -interaction=nonstopmode -halt-on-error PartB_report_HuHan.tex
```

B4 run guidance will be added alongside its implementation.
