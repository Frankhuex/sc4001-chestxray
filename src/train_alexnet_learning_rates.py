"""B1 learning-rate experiments. Run from the repository root."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
from pathlib import Path
import time
from collections.abc import Mapping, Sequence
from typing import TextIO, TypeAlias, TypedDict, cast
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MetricRow: TypeAlias = dict[str, float | int]
FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int64]
TensorLoader: TypeAlias = DataLoader[tuple[torch.Tensor, ...]]
ConfigValue: TypeAlias = str | int | float | list[int]


class Checkpoint(TypedDict):
    model_state_dict: dict[str, torch.Tensor]
    lr: float
    epoch: int
    mean: float
    std: float
    seed: int
    metrics: MetricRow


class AlexNetLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2, bias=False), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier: nn.Sequential = nn.Sequential(nn.Flatten(), nn.Linear(64*4*4, 64), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)

def evaluate(model: nn.Module, loader: TensorLoader, criterion: nn.BCEWithLogitsLoss) -> MetricRow:
    model.eval()
    total_loss: float = 0.0
    targets: list[IntArray] = []
    predictions: list[IntArray] = []
    x: torch.Tensor
    y: torch.Tensor
    p: IntArray
    with torch.no_grad():
        for x, y in loader:
            logits: torch.Tensor = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            targets.append(y.numpy().astype(int))
            predictions.append((logits >= 0).numpy().astype(int))
    target_array: IntArray = np.concatenate(targets)
    p = np.concatenate(predictions)
    tp: int = int(((target_array == 1) & (p == 1)).sum())
    tn: int = int(((target_array == 0) & (p == 0)).sum())
    fp: int = int(((target_array == 0) & (p == 1)).sum())
    fn: int = int(((target_array == 1) & (p == 0)).sum())
    accuracy: float = (tp + tn) / len(target_array)
    return dict(val_loss=total_loss/len(target_array), accuracy=accuracy, classification_error=1-accuracy,
                sensitivity=tp/(tp+fn), specificity=tn/(tn+fp), tp=tp, tn=tn, fp=fp, fn=fn)

def write_csv(path: Path, rows: Sequence[Mapping[str, float | int]]) -> None:
    f: TextIO
    with path.open('w', newline='') as f:
        writer: csv.DictWriter[str] = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def evaluate_test_checkpoint(data_path: Path, out: Path) -> None:
    """Evaluate the validation-selected model once on the held-out test set."""
    f: TextIO
    with (out / 'summary.csv').open(newline='') as f:
        candidates: list[dict[str, str]] = list(csv.DictReader(f))
    selected: dict[str, str] = min(
        candidates, key=lambda candidate: (float(candidate['classification_error']),
                                           float(candidate['val_loss'])))
    lr: float = float(selected['learning_rate'])
    checkpoint_path: Path = out / f'lr_{lr:g}_best.pt'
    checkpoint: Checkpoint = cast(Checkpoint, torch.load(
        checkpoint_path, map_location='cpu', weights_only=True))
    model: AlexNetLike = AlexNetLike()
    model.load_state_dict(checkpoint['model_state_dict'])
    data: np.lib.npyio.NpzFile
    with np.load(data_path) as data:
        images: FloatArray = data['test_images'].astype(np.float32) / 255.0
        labels: FloatArray = data['test_labels'].astype(np.float32)
    dataset: TensorDataset = TensorDataset(
        torch.from_numpy((images - checkpoint['mean']) / checkpoint['std']).unsqueeze(1),
        torch.from_numpy(labels))
    loader: TensorLoader = DataLoader(dataset, batch_size=128, shuffle=False)
    metrics: MetricRow = evaluate(model, loader, nn.BCEWithLogitsLoss())
    metrics['test_loss'] = metrics.pop('val_loss')
    result: dict[str, str | float | int] = dict(
        checkpoint=str(checkpoint_path), learning_rate=lr,
        epoch=checkpoint['epoch'], test_count=len(dataset),
        threshold=0.5, **metrics)
    (out / 'test_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    write_csv(out / 'test_metrics.csv', [dict(
        learning_rate=lr, epoch=checkpoint['epoch'], test_count=len(dataset), **metrics)])
    print(json.dumps(result, indent=2))


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--data', default='chestxray_binary_64.npz')
    parser.add_argument('--output', default='outputs/b1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--evaluate-only', action='store_true',
                        help='evaluate the validation-selected checkpoint on the test set')
    args: argparse.Namespace = parser.parse_args()
    torch.set_num_threads(args.threads)
    out: Path = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        evaluate_test_checkpoint(Path(args.data), out)
        return
    data: np.lib.npyio.NpzFile
    with np.load(args.data) as data:
        # Only train and validation arrays are read; test data are reserved for B1(d).
        train_x: FloatArray = data['train_images'].astype(np.float32) / 255.0
        val_x: FloatArray = data['val_images'].astype(np.float32) / 255.0
        train_y: FloatArray = data['train_labels'].astype(np.float32)
        val_y: FloatArray = data['val_labels'].astype(np.float32)
    mean: float = float(train_x.mean(dtype=np.float64))
    std: float = float(train_x.std(dtype=np.float64))
    train_ds: TensorDataset = TensorDataset(torch.from_numpy((train_x-mean)/std).unsqueeze(1), torch.from_numpy(train_y))
    val_ds: TensorDataset = TensorDataset(torch.from_numpy((val_x-mean)/std).unsqueeze(1), torch.from_numpy(val_y))
    val_loader: TensorLoader = DataLoader(val_ds, batch_size=128, shuffle=False)
    config: dict[str, ConfigValue] = dict(seed=args.seed, optimizer='Adam', batch_size=128, max_epochs=20, patience=4,
                  min_delta=1e-4, early_stopping_metric='validation loss',
                  best_epoch_rule='lowest validation classification error; ties broken by validation loss, then earlier epoch',
                  train_mean=mean, train_std=std, standard_deviation_ddof=0,
                  train_count=len(train_y), val_count=len(val_y),
                  train_label_counts=np.bincount(train_y.astype(int)).tolist(),
                  val_label_counts=np.bincount(val_y.astype(int)).tolist(),
                  positive_label=1, prediction_threshold=0.5, device='cpu', threads=args.threads,
                  convolution_padding='same spatial dimensions (2 for 5x5, 1 for 3x3)',
                  dropout='p=0.5 after each hidden fully connected ReLU',
                  output='single logit; BCEWithLogitsLoss; no output ReLU',
                  python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__)
    histories: dict[str, list[MetricRow]] = {}
    summary: list[MetricRow] = []
    lr: float
    epoch: int
    x: torch.Tensor
    y: torch.Tensor
    best_row: MetricRow
    metric: str
    ylabel: str
    name: str
    fig: Figure
    ax: Axes
    lr_label: str
    rows: list[MetricRow]
    for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        model: AlexNetLike = AlexNetLike()
        generator: torch.Generator = torch.Generator().manual_seed(args.seed)
        loader: TensorLoader = DataLoader(train_ds, batch_size=128, shuffle=True, generator=generator)
        optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        history: list[MetricRow] = []
        best_key: tuple[float | int, float | int] = (float('inf'), float('inf'))
        stop_best: float = float('inf')
        stale: int = 0
        started: float = time.perf_counter()
        for epoch in range(1, 21):
            model.train()
            train_loss: float = 0.0
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss: torch.Tensor = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*len(y)
            metrics: MetricRow = evaluate(model, val_loader, criterion)
            row: MetricRow = dict(epoch=epoch, train_loss=train_loss/len(train_ds), **metrics)
            history.append(row)
            key: tuple[float | int, float | int] = (metrics['classification_error'], metrics['val_loss'])
            if key < best_key:
                best_key, best_row = key, row.copy()
                torch.save(dict(model_state_dict=model.state_dict(), lr=lr, epoch=epoch,
                                mean=mean, std=std, seed=args.seed, metrics=metrics), out/f'lr_{lr:g}_best.pt')
            if metrics['val_loss'] < stop_best-1e-4:
                stop_best, stale = float(metrics['val_loss']), 0
            else:
                stale += 1
            print(f'lr={lr:g} epoch={epoch:02d} val_loss={metrics["val_loss"]:.6f} accuracy={metrics["accuracy"]:.6f} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            if stale >= 4:
                break
        histories[f'{lr:g}'] = history
        write_csv(out/f'lr_{lr:g}_history.csv', history)
        summary.append(dict(learning_rate=lr, best_epoch=best_row['epoch'], epochs_run=len(history),
                            **{k:v for k,v in best_row.items() if k not in ('epoch', 'train_loss')}))
        write_csv(out/'summary.csv', summary)
    config['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (out/'histories.json').write_text(json.dumps(histories, indent=2)+'\n')
    for metric, ylabel, name in [('val_loss', 'Validation binary cross-entropy loss', 'validation_loss'),
                                  ('accuracy', 'Validation accuracy', 'validation_accuracy')]:
        fig, ax = plt.subplots(figsize=(6.4, 3.3), layout='constrained')
        for lr_label, rows in histories.items():
            ax.plot([r['epoch'] for r in rows], [r[metric] for r in rows], marker='o', markersize=3, label=f'lr = {lr_label}')
        ax.set(xlabel='Epoch', ylabel=ylabel, xticks=range(1, 21))
        if metric == 'accuracy':
            ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(out/f'{name}.pdf')
        fig.savefig(out/f'{name}.png', dpi=180)
        plt.close(fig)
    print('Completed all five learning rates.', flush=True)

if __name__ == '__main__':
    main()
