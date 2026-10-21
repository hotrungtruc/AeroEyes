"""Distributed training entry point for AeroEyes on Kaggle or Google Colab.

Launch with torchrun when more than one GPU is available:

    torchrun --standalone --nproc_per_node=2 train_ddp.py \
        --data-root /kaggle/input/aeroeyes-dataset --epochs 40 --batch-size 2 \
        --output-dir /kaggle/working/aeroeyes-ddp --export-onnx

With one GPU (or a normal ``python train_ddp.py`` invocation), the same file
falls back to a regular single-process training run.  The model, dataset and
loss are shared with train.py so both entry points train the same network.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import (
    AeroEyesDataset,
    detector_loss,
    export_onnx,
    load_records,
    seed_everything,
)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CSPBottleneck(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = max(8, int(channels * expansion))
        self.left = ConvNormAct(channels, hidden, 1)
        self.right = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(hidden * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.merge(torch.cat((self.left(x), self.right(x)), dim=1)))


class ConvNeXtFemtoBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, channels * 4, 1)
        self.project = nn.Conv2d(channels * 4, channels, 1)
        self.scale = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.project(F.gelu(self.expand(self.norm(self.depthwise(x)))))
        return x + update * self.scale.view(1, -1, 1, 1)


class HybridBackbone(nn.Module):
    channels: Tuple[int, int, int] = (48, 96, 160)

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvNormAct(3, 32, 2)
        self.stage1 = nn.Sequential(ConvNormAct(32, 48, 2), CSPBottleneck(48))
        self.stage2 = nn.Sequential(ConvNormAct(48, 96, 2), CSPBottleneck(96), ConvNeXtFemtoBlock(96))
        self.stage3 = nn.Sequential(ConvNormAct(96, 160, 2), CSPBottleneck(160), ConvNeXtFemtoBlock(160))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return [p3, p4, p5]


class GlobalLocalPrototype(nn.Module):
    def __init__(self, channels: Sequence[int], prototype_dim: int = 96) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(c, prototype_dim, 1) for c in channels)
        self.relation = nn.Sequential(
            nn.Linear(prototype_dim, prototype_dim // 2),
            nn.SiLU(),
            nn.Linear(prototype_dim // 2, 1),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        pooled = [
            self.projections[i](feature.mean(dim=1)).mean(dim=(-2, -1))
            for i, feature in enumerate(features)
        ]
        level_stack = torch.stack(pooled, dim=1)
        weights = torch.softmax(self.relation(level_stack).squeeze(-1), dim=1)
        prototype = (level_stack * weights.unsqueeze(-1)).sum(dim=1)
        support_maps = [self.projections[i](features[i].mean(dim=1)) for i in range(len(features))]
        return prototype, [value + prototype[:, :, None, None] for value in support_maps]


class PrototypeRelationFusion(nn.Module):
    def __init__(self, channels: Sequence[int], prototype_dim: int = 96) -> None:
        super().__init__()
        self.query_projections = nn.ModuleList(nn.Conv2d(c, prototype_dim, 1) for c in channels)
        self.prototype_projections = nn.ModuleList(nn.Conv2d(prototype_dim, c, 1) for c in channels)

    def forward(
        self, query_features: Sequence[torch.Tensor], support_maps: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        fused: List[torch.Tensor] = []
        for index, (query, support) in enumerate(zip(query_features, support_maps)):
            query_embedding = self.query_projections[index](query)
            if support.shape[-2:] != query_embedding.shape[-2:]:
                support = F.interpolate(support, size=query_embedding.shape[-2:], mode="bilinear",
                                        align_corners=False)
            relation = (
                F.normalize(query_embedding, dim=1) * F.normalize(support, dim=1)
            ).sum(dim=1, keepdim=True)
            support_bias = self.prototype_projections[index](support)
            if support_bias.shape[-2:] != query.shape[-2:]:
                support_bias = F.interpolate(support_bias, size=query.shape[-2:], mode="bilinear",
                                             align_corners=False)
            fused.append(query + query * relation.sigmoid() + support_bias)
        return fused


class AirDetLiteHybridDetector(nn.Module):
    """AirDet-inspired multi-scale prototype detector with a lightweight backbone."""

    def __init__(self, prototype_dim: int = 96) -> None:
        super().__init__()
        self.backbone = HybridBackbone()
        channels = self.backbone.channels
        self.prototype = GlobalLocalPrototype(channels, prototype_dim)
        self.fusion = PrototypeRelationFusion(channels, prototype_dim)
        self.fpn_projections = nn.ModuleList(nn.Conv2d(c, channels[0], 1) for c in channels[1:])
        self.level_heads = nn.ModuleList(
            nn.Sequential(ConvNormAct(c, c), nn.Conv2d(c, 5, 1)) for c in channels
        )

    def forward(self, support: torch.Tensor, search: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if support.ndim == 4:
            support = support.unsqueeze(1)
        if support.ndim != 5 or search.ndim != 4:
            raise ValueError("support must be [B,3,H,W] or [B,K,3,H,W]; search must be [B,3,H,W]")
        batch, shots, channels, height, width = support.shape
        support_features_flat = self.backbone(support.reshape(batch * shots, channels, height, width))
        support_features = [
            value.reshape(batch, shots, value.shape[1], value.shape[2], value.shape[3])
            for value in support_features_flat
        ]
        query_features = self.backbone(search)
        _, support_maps = self.prototype(support_features)
        fused = self.fusion(query_features, support_maps)
        p3_size = fused[0].shape[-2:]
        fused_map = fused[0]
        for projection, value in zip(self.fpn_projections, fused[1:]):
            fused_map = fused_map + F.interpolate(projection(value), size=p3_size,
                                                   mode="bilinear", align_corners=False)
        outputs = [self.level_heads[0](fused_map)]
        outputs.extend(self.level_heads[i](value) for i, value in enumerate(fused[1:], start=1))
        objectness, box = outputs[0][:, 0:1], outputs[0][:, 1:5]
        for output in outputs[1:]:
            resized = F.interpolate(output, size=p3_size, mode="bilinear", align_corners=False)
            objectness = objectness + resized[:, 0:1]
            box = box + resized[:, 1:5]
        return objectness / len(outputs), box / len(outputs)


def distributed_context(requested_device: str) -> Tuple[int, int, int, torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = requested_device != "cpu" and torch.cuda.is_available()
    if requested_device == "cuda" and not use_cuda:
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    if use_cuda:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank, device, world_size > 1


def reduce_metrics(
    totals: Dict[str, float], samples: int, device: torch.device, distributed: bool
) -> Dict[str, float]:
    values = torch.tensor(
        [totals["loss"], totals["bce"], totals["nwd"], float(samples)],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = max(float(values[3]), 1.0)
    return {"loss": float(values[0] / count), "bce": float(values[1] / count),
            "nwd": float(values[2] / count)}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    image_size: int,
    distributed: bool,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "bce": 0.0, "nwd": 0.0}
    samples = 0
    for query, search, boxes in loader:
        query = query.to(device, non_blocking=True)
        search = search.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            objectness, box = model(query, search)
            loss, metrics = detector_loss(objectness, box, boxes, image_size)
        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        for key in totals:
            totals[key] += metrics[key] * query.shape[0]
        samples += query.shape[0]
    return reduce_metrics(totals, samples, device, distributed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Dataset root containing train/annotations and train/samples")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/aeroeyes-ddp"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-GPU batch size, not the global batch size")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.image_size < 64 or args.batch_size < 1:
        raise ValueError("epochs, batch-size and image-size must be positive")
    rank, world_size, _, device, distributed = distributed_context(args.device)
    seed_everything(args.seed + rank)
    data_root = args.data_root.resolve()
    records, video_ids = load_records(data_root)
    random.Random(args.seed).shuffle(video_ids)
    val_count = max(1, int(round(len(video_ids) * args.val_fraction)))
    val_ids, train_ids = video_ids[:val_count], video_ids[val_count:]
    train_set = AeroEyesDataset(data_root, records, train_ids, args.image_size, augment=True)
    val_set = AeroEyesDataset(data_root, records, val_ids, args.image_size, augment=False)
    if args.max_train_samples:
        train_set.records = train_set.records[:args.max_train_samples]
    if args.max_val_samples:
        val_set.records = val_set.records[:args.max_val_samples]

    train_sampler = DistributedSampler(train_set, world_size, rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, world_size, rank, shuffle=False) if distributed else None
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_set, sampler=train_sampler, shuffle=train_sampler is None,
                              **loader_kwargs)
    val_loader = DataLoader(val_set, sampler=val_sampler, shuffle=False, **loader_kwargs)

    model = AirDetLiteHybridDetector().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 50_000_000:
        raise RuntimeError("Model has %d parameters; Jetson limit is 50 million" % parameter_count)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint["model"]) if distributed else model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("val_loss", best_val))

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print("world_size=%d device=%s parameters=%d global_batch_size=%d" %
              (world_size, device, parameter_count, args.batch_size * world_size))
        print("train_videos=%s val_videos=%s train_boxes=%d val_boxes=%d" %
              (",".join(train_ids), ",".join(val_ids), len(train_set), len(val_set)))

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        started = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device,
                                  args.image_size, distributed)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, None, device,
                                    args.image_size, distributed)
        scheduler.step()
        if rank == 0:
            print("epoch=%03d train_loss=%.5f val_loss=%.5f train_nwd=%.5f val_nwd=%.5f time=%.1fs" %
                  (epoch, train_metrics["loss"], val_metrics["loss"], train_metrics["nwd"],
                   val_metrics["nwd"], time.time() - started))
            state = model.module.state_dict() if distributed else model.state_dict()
            checkpoint = {"model": state, "epoch": epoch, "val_loss": val_metrics["loss"],
                          "image_size": args.image_size, "train_ids": train_ids, "val_ids": val_ids}
            torch.save(checkpoint, args.output_dir / "last.pt")
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(checkpoint, args.output_dir / "best.pt")
        if distributed:
            dist.barrier()

    if rank == 0 and args.export_onnx:
        best = torch.load(args.output_dir / "best.pt", map_location=device)
        export_model = AirDetLiteHybridDetector().to(device)
        export_model.load_state_dict(best["model"])
        export_onnx(export_model, args.output_dir / "aeroeyes_detector.onnx", args.image_size, device)
        print("exported=%s" % (args.output_dir / "aeroeyes_detector.onnx"))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
