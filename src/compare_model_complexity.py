"""Compare saved test metrics and inference complexity for the three CNNs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TextIO, TypedDict

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from train_alexnet_learning_rates import AlexNetLike
from train_vgg_batch_sizes import VGGNetLike
from train_resnet_classifier import ResNetLike


class ComparisonRow(TypedDict):
    model: str
    accuracy_percent: float
    sensitivity_percent: float
    specificity_percent: float
    classification_error_percent: float
    trainable_parameters: int
    macs: int
    flops: int


def measure_complexity(model: nn.Module) -> tuple[int, int, int]:
    """Count convolution/linear MACs for one 1×64×64 image; FLOPs = 2×MACs."""
    macs: int = 0
    handles: list[RemovableHandle] = []

    def count_layer(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    layer: nn.Module
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(count_layer))
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, 1, 64, 64))
    finally:
        handle: RemovableHandle
        for handle in handles:
            handle.remove()
    parameters: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return parameters, macs, 2 * macs


def main() -> None:
    torch.set_num_threads(6)
    rows: list[ComparisonRow] = []
    name: str
    part: str
    model: nn.Module
    parameters: int
    macs: int
    flops: int
    for name, part, model in [('AlexNet-like', 'b1', AlexNetLike()),
                              ('VGG-like', 'b2', VGGNetLike()),
                              ('ResNet-like', 'b3', ResNetLike())]:
        metrics: dict[str, str | int | float] = json.loads(Path(f'outputs/{part}/test_metrics.json').read_text())
        parameters, macs, flops = measure_complexity(model)
        rows.append(dict(model=name, accuracy_percent=100*float(metrics['accuracy']),
                         sensitivity_percent=100*float(metrics['sensitivity']),
                         specificity_percent=100*float(metrics['specificity']),
                         classification_error_percent=100*float(metrics['classification_error']),
                         trainable_parameters=parameters, macs=macs, flops=flops))
    out: Path = Path('outputs/b3')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'architecture_comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    f: TextIO
    with (out / 'architecture_comparison.csv').open('w', newline='') as f:
        writer: csv.DictWriter[str] = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
