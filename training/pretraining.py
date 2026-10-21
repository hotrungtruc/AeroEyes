"""Context pretraining for the AeroEyes AirDet-lite hybrid detector.

This script pretrains the same class-agnostic objectness and localization
branches used by ``main_training.py`` on public UAV datasets before target
training. It supports common COCO JSON, YOLO TXT, VisDrone TXT and UAVDT GT
layouts without downloading data or sending images to an external service.

Example (Kaggle/Colab):
    python training/pretraining.py \
      --dataset-root uavdt=/kaggle/input/uavdt-det \
      --dataset-root visdrone=/kaggle/input/visdrone \
      --dataset-root anti_uav=/kaggle/input/anti-uav \
      --dataset-root visdrone_video=/kaggle/input/visdrone-videos \
      --epochs 20 --output-dir /kaggle/working/aeroeyes-pretrain

Dataset sources referenced by the project:
UAVDT, VisDrone DET/VID and Anti-UAV. The dataset archives must be downloaded
and accepted under their respective licenses by the user first.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_training import AirDetLiteHybridDetector
from train import detector_loss, seed_everything


@dataclass(frozen=True)
class ContextRecord:
    image: Path
    box: Tuple[float, float, float, float]
    source: str


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(root: Path) -> List[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def parse_yolo_file(label: Path, image_width: int, image_height: int) -> List[Tuple[float, float, float, float]]:
    boxes = []
    for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            _, cx, cy, width, height = map(float, fields[:5])
        except ValueError:
            continue
        boxes.append(((cx - width / 2) * image_width, (cy - height / 2) * image_height,
                      (cx + width / 2) * image_width, (cy + height / 2) * image_height))
    return boxes


def parse_visdrone_file(label: Path) -> List[Tuple[float, float, float, float]]:
    boxes = []
    for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.replace(" ", ",").split(",")
        if len(fields) < 6:
            continue
        try:
            x, y, width, height = [float(value) for value in fields[:4]]
            score = float(fields[4])
            category = int(float(fields[5]))
        except ValueError:
            continue
        if score > 0 and category not in (0, 11):
            boxes.append((x, y, x + width, y + height))
    return boxes


def parse_uavdt_file(label: Path) -> List[Tuple[float, float, float, float]]:
    boxes = []
    for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.replace(" ", ",").split(",")
        if len(fields) < 6:
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            continue
        # UAVDT gt commonly stores frame,id,x,y,w,h,...; accept both that
        # format and plain x,y,w,h rows.
        offset = 2 if len(values) >= 8 else 0
        x, y, width, height = values[offset:offset + 4]
        if width > 1 and height > 1:
            boxes.append((x, y, x + width, y + height))
    return boxes


def load_coco(root: Path, source: str) -> List[ContextRecord]:
    records: List[ContextRecord] = []
    for annotation in root.rglob("*.json"):
        try:
            payload = json.loads(annotation.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or not {"images", "annotations"} <= set(payload):
            continue
        images = {item["id"]: item for item in payload["images"] if "id" in item and "file_name" in item}
        for item in payload["annotations"]:
            image = images.get(item.get("image_id"))
            bbox = item.get("bbox")
            if image is None or not bbox or len(bbox) < 4:
                continue
            path = root / image["file_name"]
            if not path.exists():
                path = annotation.parent / image["file_name"]
            if path.exists() and bbox[2] > 0 and bbox[3] > 0:
                records.append(ContextRecord(path, (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]), source))
    return records


def load_directory(root: Path, source: str) -> List[ContextRecord]:
    records: List[ContextRecord] = []
    for image in image_files(root):
        try:
            with Image.open(image) as opened:
                width, height = opened.size
        except OSError:
            continue
        candidates = [
            image.with_suffix(".txt"),
            image.parent / "annotations" / (image.stem + ".txt"),
            image.parent.parent / "annotations" / (image.stem + ".txt"),
            root / "gt" / (image.stem + ".txt"),
        ]
        label = next((candidate for candidate in candidates if candidate.exists()), None)
        if label is None:
            continue
        text = label.read_text(encoding="utf-8", errors="ignore")
        first_fields = next((line.replace(" ", ",").split(",") for line in text.splitlines() if line.strip()), [])
        if len(first_fields) >= 8:
            boxes = parse_uavdt_file(label)
        elif len(first_fields) >= 6:
            boxes = parse_visdrone_file(label)
        else:
            boxes = parse_yolo_file(label, width, height)
        records.extend(ContextRecord(image, box, source) for box in boxes
                       if box[2] > box[0] and box[3] > box[1])
    return records


def discover_records(datasets: Sequence[str]) -> List[ContextRecord]:
    records: List[ContextRecord] = []
    for item in datasets:
        if "=" not in item:
            raise ValueError("--dataset-root must use NAME=PATH, for example uavdt=/data/UAVDT")
        name, raw_path = item.split("=", 1)
        root = Path(raw_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError("Dataset root does not exist: " + str(root))
        found = load_coco(root, name)
        found.extend(load_directory(root, name))
        if not found:
            raise ValueError("No supported annotated images found in " + str(root))
        records.extend(found)
        print("dataset=%s records=%d root=%s" % (name, len(found), root))
    if not records:
        raise ValueError("No context datasets were provided")
    return records


class ContextDataset(Dataset):
    def __init__(self, records: Sequence[ContextRecord], image_size: int, augment: bool) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]
        image = Image.open(record.image).convert("RGB")
        width, height = image.size
        box = np.array(record.box, dtype=np.float32)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            box[[0, 2]] = width - box[[2, 0]]
        if self.augment:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        box[[0, 2]] *= self.image_size / max(width, 1)
        box[[1, 3]] *= self.image_size / max(height, 1)
        box = torch.from_numpy(np.clip(box, 0, self.image_size - 1).copy())
        search = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        # Context pretraining is class-agnostic. A lightly perturbed view is
        # used as support so the relation branch learns generic visual cues.
        support = search.clone()
        if self.augment:
            support = (support * random.uniform(0.9, 1.1)).clamp(0, 1)
        return support, search, box


def context_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer],
                  scaler: Optional[torch.cuda.amp.GradScaler], device: torch.device,
                  image_size: int, distributed: bool) -> Dict[str, float]:
    model.train(optimizer is not None)
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for support, search, boxes in loader:
        support, search, boxes = support.to(device), search.to(device), boxes.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=amp):
            objectness, box = model(support, search)
            loss, metrics = detector_loss(objectness, box, boxes, image_size)
        if optimizer is not None:
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        totals[:3] += torch.tensor([metrics["loss"], metrics["bce"], metrics["nwd"]],
                                   dtype=torch.float64, device=device) * support.shape[0]
        totals[3] += support.shape[0]
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(float(totals[3]), 1.0)
    return {"loss": float(totals[0] / count), "bce": float(totals[1] / count),
            "nwd": float(totals[2] / count)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", required=True,
                        help="Named dataset root: NAME=PATH. Repeat for UAVDT, VisDrone, Anti-UAV, etc.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/aeroeyes-pretrain"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    if world_size > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo", init_method="env://")
        if use_cuda:
            torch.cuda.set_device(device)
    seed_everything(args.seed + rank)
    records = discover_records(args.dataset_root)
    random.Random(args.seed).shuffle(records)
    split = int(len(records) * args.val_fraction)
    if len(records) > 1:
        split = max(1, min(split, len(records) - 1))
        val_records, train_records = records[:split], records[split:]
    else:
        # Keep a one-record fixture runnable for notebook smoke tests.
        val_records = list(records)
        train_records = list(records)
    if args.max_samples:
        train_records = train_records[:args.max_samples]
        val_records = val_records[:max(1, min(args.max_samples, len(val_records)))]
    train_set = ContextDataset(train_records, args.image_size, True)
    val_set = ContextDataset(val_records, args.image_size, False)
    train_sampler = DistributedSampler(train_set, world_size, rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, world_size, rank, shuffle=False) if world_size > 1 else None
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
                      "pin_memory": use_cuda, "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_set, sampler=train_sampler, shuffle=train_sampler is None, **loader_options)
    val_loader = DataLoader(val_set, sampler=val_sampler, shuffle=False, **loader_options)
    model = AirDetLiteHybridDetector().to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.cuda.amp.GradScaler() if use_cuda else None
    start_epoch, best_val = 1, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        (model.module if world_size > 1 else model).load_state_dict(state["model"])
        start_epoch, best_val = state["epoch"] + 1, state.get("val_loss", best_val)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print("pretraining records=%d train=%d val=%d world_size=%d device=%s" %
              (len(records), len(train_set), len(val_set), world_size, device))
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        started = time.time()
        train_metrics = context_epoch(model, train_loader, optimizer, scaler, device,
                                      args.image_size, world_size > 1)
        with torch.no_grad():
            val_metrics = context_epoch(model, val_loader, None, None, device,
                                        args.image_size, world_size > 1)
        scheduler.step()
        if rank == 0:
            state = (model.module if world_size > 1 else model).state_dict()
            checkpoint = {"model": state, "epoch": epoch, "val_loss": val_metrics["loss"],
                          "image_size": args.image_size, "pretraining": True}
            torch.save(checkpoint, args.output_dir / "last.pt")
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(checkpoint, args.output_dir / "best.pt")
            print("epoch=%03d train_loss=%.5f val_loss=%.5f train_nwd=%.5f val_nwd=%.5f time=%.1fs" %
                  (epoch, train_metrics["loss"], val_metrics["loss"], train_metrics["nwd"],
                   val_metrics["nwd"], time.time() - started))
        if world_size > 1:
            dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
