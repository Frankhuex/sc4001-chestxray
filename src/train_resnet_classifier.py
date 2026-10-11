"""Train the fixed compact ResNet-like classifier and evaluate its saved checkpoint."""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_alexnet_learning_rates import ConfigValue, FloatArray, MetricRow, TensorLoader, evaluate, write_csv
from train_vgg_batch_sizes import Checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.branch: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU())
        self.activation: nn.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.branch(x) + self.shortcut(x))


class ResNetLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.blocks: nn.Sequential = nn.Sequential(
            ResidualBlock(16, 16), ResidualBlock(16, 32, 2), ResidualBlock(32, 64, 2))
        self.pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier: nn.Linear = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.blocks(self.stem(x))
        return self.classifier(self.pool(features).flatten(1)).squeeze(1)


def evaluate_test_checkpoint(data_path: Path, out: Path) -> None:
    checkpoint_path: Path = out / 'best.pt'
    checkpoint: Checkpoint = cast(Checkpoint, torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model: ResNetLike = ResNetLike()
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
    result: dict[str, str | int | float] = dict(
        checkpoint=str(checkpoint_path), learning_rate=checkpoint['lr'], batch_size=checkpoint['batch_size'],
        epoch=checkpoint['epoch'], test_count=len(dataset), threshold=0.5, **metrics)
    (out / 'test_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    write_csv(out / 'test_metrics.csv', [dict(learning_rate=checkpoint['lr'], batch_size=checkpoint['batch_size'],
                                             epoch=checkpoint['epoch'], test_count=len(dataset), **metrics)])
    print(json.dumps(result, indent=2), flush=True)


def plot_training_curves(history: list[MetricRow], out: Path, best_epoch: int) -> None:
    epochs: list[int] = [int(row['epoch']) for row in history]
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6.4, 3.3), layout='constrained')
    ax.plot(epochs, [row['train_loss'] for row in history], marker='o', markersize=3, label='Training')
    ax.plot(epochs, [row['val_loss'] for row in history], marker='o', markersize=3, label='Validation')
    ax.axvline(best_epoch, color='gray', linestyle='--', label=f'Selected epoch {best_epoch}')
    ax.set(xlabel='Epoch', ylabel='Binary cross-entropy loss', xticks=epochs)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out / 'training_curves.pdf')
    fig.savefig(out / 'training_curves.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--data', default='chestxray_binary_64.npz')
    parser.add_argument('--output', default='outputs/b3')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--evaluate-only', action='store_true')
    args: argparse.Namespace = parser.parse_args()
    torch.set_num_threads(args.threads)
    out: Path = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        evaluate_test_checkpoint(Path(args.data), out)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    data: np.lib.npyio.NpzFile
    with np.load(args.data) as data:
        train_x: FloatArray = data['train_images'].astype(np.float32) / 255.0
        val_x: FloatArray = data['val_images'].astype(np.float32) / 255.0
        train_y: FloatArray = data['train_labels'].astype(np.float32)
        val_y: FloatArray = data['val_labels'].astype(np.float32)
    mean: float = float(train_x.mean(dtype=np.float64))
    std: float = float(train_x.std(dtype=np.float64))
    train_ds: TensorDataset = TensorDataset(torch.from_numpy((train_x-mean)/std).unsqueeze(1), torch.from_numpy(train_y))
    val_ds: TensorDataset = TensorDataset(torch.from_numpy((val_x-mean)/std).unsqueeze(1), torch.from_numpy(val_y))
    generator: torch.Generator = torch.Generator().manual_seed(args.seed)
    loader: TensorLoader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader: TensorLoader = DataLoader(val_ds, batch_size=128, shuffle=False)
    model: ResNetLike = ResNetLike()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    config: dict[str, ConfigValue] = dict(
        seed=args.seed, optimizer='Adam', learning_rate=args.learning_rate, batch_size=args.batch_size,
        max_epochs=20, patience=4, min_delta=1e-4, early_stopping_metric='validation loss',
        best_epoch_rule='lowest validation classification error; ties: lower loss, then earlier epoch',
        train_mean=mean, train_std=std, standard_deviation_ddof=0, train_count=len(train_ds), val_count=len(val_ds),
        device='cpu', threads=args.threads, evaluation_batch_size=128, positive_label=1, prediction_threshold=0.5,
        convolution_padding='SAME; padding 1 for 3x3, padding 0 for 1x1',
        activations='BatchNorm and ReLU after every convolution (including projections); ReLU after residual addition',
        dropout='none, following supplementary slides', output='single logit; BCEWithLogitsLoss',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    history: list[MetricRow] = []
    best_row: MetricRow = {}
    best_key: tuple[float | int, float | int] = (float('inf'), float('inf'))
    stop_best: float = float('inf')
    stale: int = 0
    started: float = time.perf_counter()
    epoch: int
    x: torch.Tensor
    y: torch.Tensor
    for epoch in range(1, 21):
        model.train()
        train_loss: float = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss: torch.Tensor = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        metrics: MetricRow = evaluate(model, val_loader, criterion)
        row: MetricRow = dict(epoch=epoch, train_loss=train_loss/len(train_ds), **metrics)
        history.append(row)
        key: tuple[float | int, float | int] = (metrics['classification_error'], metrics['val_loss'])
        if key < best_key:
            best_key, best_row = key, row.copy()
            checkpoint: Checkpoint = dict(model_state_dict=model.state_dict(), lr=args.learning_rate,
                                          batch_size=args.batch_size, epoch=epoch, mean=mean, std=std,
                                          seed=args.seed, metrics=metrics)
            torch.save(checkpoint, out / 'best.pt')
        if metrics['val_loss'] < stop_best - 1e-4:
            stop_best, stale = float(metrics['val_loss']), 0
        else:
            stale += 1
        write_csv(out / 'history.csv', history)
        print(f'epoch={epoch:02d} val_loss={metrics["val_loss"]:.6f} accuracy={metrics["accuracy"]:.6f} elapsed={time.perf_counter()-started:.1f}s', flush=True)
        if stale >= 4:
            break
    summary: MetricRow = dict(learning_rate=args.learning_rate, batch_size=args.batch_size,
                              best_epoch=best_row['epoch'], epochs_run=len(history),
                              **{k: v for k, v in best_row.items() if k not in ('epoch', 'train_loss')})
    write_csv(out / 'summary.csv', [summary])
    (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    plot_training_curves(history, out, int(best_row['epoch']))
    print('Completed ResNet training.', flush=True)


if __name__ == '__main__':
    main()
