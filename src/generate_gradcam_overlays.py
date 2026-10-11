"""Generate pneumonia Grad-CAM overlays for the three saved CNN classifiers."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from captum.attr import LayerAttribution, LayerGradCam

from train_alexnet_learning_rates import AlexNetLike, FloatArray
from train_vgg_batch_sizes import Checkpoint, VGGNetLike
from train_resnet_classifier import ResNetLike


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--data', default='chestxray_binary_64.npz')
    parser.add_argument('--output', default='outputs/b4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=6)
    args: argparse.Namespace = parser.parse_args()
    torch.set_num_threads(args.threads)
    out: Path = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data: np.lib.npyio.NpzFile
    with np.load(args.data) as data:
        images: FloatArray = data['test_images'].astype(np.float32) / 255.0
        labels: NDArray[np.int64] = data['test_labels']
    indices: NDArray[np.int64] = np.random.default_rng(args.seed).choice(np.flatnonzero(labels == 1), size=5, replace=False)
    selected: FloatArray = images[indices]
    alexnet: AlexNetLike = AlexNetLike()
    vgg: VGGNetLike = VGGNetLike()
    resnet: ResNetLike = ResNetLike()
    models: list[tuple[str, nn.Module, nn.Module, Path]] = [
        ('AlexNet', alexnet, alexnet.features[8], Path('outputs/b1/lr_0.001_best.pt')),
        ('VGG', vgg, vgg.features[17], Path('outputs/b2/batch_16_best.pt')),
        ('ResNet', resnet, resnet.blocks[2].branch[3], Path('outputs/b3/best.pt'))]
    raw_maps: dict[str, FloatArray] = {}
    upsampled_maps: dict[str, FloatArray] = {}
    normalized_maps: dict[str, FloatArray] = {}
    probabilities: dict[str, list[float]] = {}
    details: list[dict[str, str | int | float | list[int] | list[float]]] = []
    name: str
    model: nn.Module
    layer: nn.Module
    path: Path
    for name, model, layer, path in models:
        checkpoint: Checkpoint = cast(Checkpoint, torch.load(path, map_location='cpu', weights_only=True))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        x: torch.Tensor = torch.from_numpy((selected - checkpoint['mean']) / checkpoint['std']).unsqueeze(1)
        x.requires_grad_(True)
        cam: LayerGradCam = LayerGradCam(model, layer)
        signed: torch.Tensor = cam.attribute(x, relu_attributions=False)
        attribution: torch.Tensor = signed.clamp_min(0)
        enlarged: torch.Tensor = LayerAttribution.interpolate(attribution, (64, 64), interpolate_mode='bilinear')
        raw: FloatArray = attribution.detach().numpy()[:, 0].astype(np.float32)
        values: FloatArray = enlarged.detach().numpy()[:, 0].astype(np.float32)
        minima: FloatArray = values.min(axis=(1, 2), keepdims=True)
        ranges: FloatArray = values.max(axis=(1, 2), keepdims=True) - minima
        normalized: FloatArray = np.divide(values - minima, ranges, out=np.zeros_like(values), where=ranges > 0)
        with torch.no_grad():
            probability: list[float] = torch.sigmoid(model(x)).tolist()
        raw_maps[name], upsampled_maps[name], normalized_maps[name] = raw, values, normalized
        probabilities[name] = probability
        details.append(dict(model=name, checkpoint=str(path), epoch=checkpoint['epoch'],
                            target_layer=str(layer), spatial_shape=list(raw.shape[1:]),
                            signed_map_minima=signed.detach().flatten(1).min(dim=1).values.tolist(),
                            signed_map_maxima=signed.detach().flatten(1).max(dim=1).values.tolist(),
                            training_mean=checkpoint['mean'], training_std=checkpoint['std'],
                            constant_map_indices=[int(indices[i]) for i in range(5) if ranges[i, 0, 0] == 0]))
        print(f'{name}: CAM shape {raw.shape}, pneumonia probabilities {probability}', flush=True)
    fig: Figure
    axes: NDArray[np.object_]
    ax: Axes
    fig, axes = plt.subplots(5, 4, figsize=(8.0, 10.0), layout='constrained')
    i: int
    j: int
    for i in range(5):
        axes[i, 0].imshow(selected[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Test index {int(indices[i])}', fontsize=9)
        for j, (name, model, layer, path) in enumerate(models, start=1):
            axes[i, j].imshow(selected[i], cmap='gray', vmin=0, vmax=1)
            axes[i, j].imshow(normalized_maps[name][i], cmap='jet', vmin=0, vmax=1, alpha=0.4)
            axes[i, j].set_title(f'{name}: p={probabilities[name][i]:.3f}', fontsize=9)
            single: Figure
            ax: Axes
            single, ax = plt.subplots(figsize=(3, 3), layout='constrained')
            ax.imshow(selected[i], cmap='gray', vmin=0, vmax=1)
            ax.imshow(normalized_maps[name][i], cmap='jet', vmin=0, vmax=1, alpha=0.4)
            ax.set_title(f'{name}, index {int(indices[i])}, p={probabilities[name][i]:.3f}', fontsize=9)
            ax.axis('off')
            single.savefig(out / f'index_{int(indices[i])}_{name.lower()}.png', dpi=180)
            plt.close(single)
        for ax in axes[i]:
            ax.axis('off')
    fig.savefig(out / 'gradcam_overlays.png', dpi=200)
    fig.savefig(out / 'gradcam_overlays.pdf')
    plt.close(fig)
    np.savez(out / 'gradcam_arrays.npz', test_indices=indices, images=selected,
             **{f'{name.lower()}_raw': value for name, value in raw_maps.items()},
             **{f'{name.lower()}_upsampled': value for name, value in upsampled_maps.items()},
             **{f'{name.lower()}_normalized': value for name, value in normalized_maps.items()})
    configuration: dict[str, object] = dict(seed=args.seed, split='test', positive_label=1,
        test_indices=indices.tolist(), sampling='random without replacement among pneumonia images',
        attribution_target='pneumonia logit', target_layers=details, interpolation='bilinear',
        normalization='per-image min-max after positive attribution clipping', colormap='jet', alpha=0.4,
        probabilities=probabilities, torch=torch.__version__, captum_method='LayerGradCam', threads=args.threads)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    (out / 'config.json').write_text(json.dumps(configuration, indent=2) + '\n')
    print('Completed all 15 Grad-CAM overlays.', flush=True)


if __name__ == '__main__':
    main()
