"""Train the AeroEyes query-based single-stage detector.

The repository annotations contain one target box per annotated video frame.  The
target class is intentionally not used as a classifier: the query images define
which object the detector should find.  This keeps the training objective aligned
with the query-based deployment described in README.md.

Example:
    python train.py --data-root AeroEyes_dataset --epochs 40 --device cuda

The script only depends on PyTorch, OpenCV, NumPy and Pillow, all of which are
already listed in requirements-jetson.txt (PyTorch is installed separately on
Jetson).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class BoxRecord:
    video_id: str
    frame: int
    box: Tuple[float, float, float, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rgb(path: Path, size: int, augment: bool) -> Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    if augment:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.35:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.25))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0


class AeroEyesDataset(Dataset):
    """On-demand video frame loader with one query image per training item."""

    def __init__(
        self,
        root: Path,
        records: Sequence[BoxRecord],
        video_ids: Sequence[str],
        image_size: int,
        augment: bool,
        references_per_target: int = 3,
    ) -> None:
        self.root = root
        self.records = [r for r in records if r.video_id in set(video_ids)]
        self.image_size = image_size
        self.augment = augment
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_id: Optional[str] = None
        self.references: Dict[str, List[Path]] = {}
        for video_id in video_ids:
            paths = sorted((root / "train" / "samples" / video_id / "object_images").glob("*.jpg"))
            if not paths:
                raise FileNotFoundError("No reference images found for " + video_id)
            self.references[video_id] = paths[:references_per_target]

    def __len__(self) -> int:
        return len(self.records)

    def _get_frame(self, video_id: str, frame_number: int) -> np.ndarray:
        video_path = self.root / "train" / "samples" / video_id / "drone_video.mp4"
        if self._capture_id != video_id:
            if self._capture is not None:
                self._capture.release()
            self._capture = cv2.VideoCapture(str(video_path))
            self._capture_id = video_id
        assert self._capture is not None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("Could not read frame %d from %s" % (frame_number, video_path))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]
        frame = Image.fromarray(self._get_frame(record.video_id, record.frame))
        width, height = frame.size
        box = np.array(record.box, dtype=np.float32)
        frame = frame.resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.augment and random.random() < 0.5:
            frame = frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            box[[0, 2]] = width - box[[2, 0]]
        if self.augment and random.random() < 0.25:
            frame = ImageEnhance.Brightness(frame).enhance(random.uniform(0.8, 1.2))
            frame = ImageEnhance.Contrast(frame).enhance(random.uniform(0.8, 1.2))
        box[[0, 2]] *= self.image_size / max(width, 1)
        box[[1, 3]] *= self.image_size / max(height, 1)
        box = np.clip(box, 0, self.image_size - 1)
        query_path = random.choice(self.references[record.video_id])
        query = read_rgb(query_path, self.image_size, self.augment)
        search = torch.from_numpy(np.asarray(frame, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        return query, search, torch.from_numpy(box)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FeatureEncoder(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(3, width, 2),
            ConvBNAct(width, width * 2, 2),
            ConvBNAct(width * 2, width * 4, 2),
            ConvBNAct(width * 4, width * 8, 2),
            ConvBNAct(width * 8, width * 8, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class SiameseModulatedDetector(nn.Module):
    """Compact query-conditioned detector, under the Jetson parameter budget."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(width)
        channels = width * 8
        self.head = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, 5, 1))

    def forward(self, query: Tensor, search: Tensor) -> Tuple[Tensor, Tensor]:
        query_feature = self.encoder(query)
        search_feature = self.encoder(search)
        query_embedding = F.normalize(query_feature.mean(dim=(2, 3)), dim=1)
        search_norm = F.normalize(search_feature, dim=1)
        modulation = (search_norm * query_embedding[:, :, None, None]).sum(dim=1, keepdim=True)
        fused = search_feature * (1.0 + modulation)
        output = self.head(fused)
        objectness = output[:, 0:1]
        box = output[:, 1:5]
        return objectness, box


def make_targets(boxes: Tensor, grid_h: int, grid_w: int, image_size: int) -> Tuple[Tensor, Tensor]:
    batch = boxes.shape[0]
    objectness = torch.zeros(batch, 1, grid_h, grid_w, device=boxes.device)
    target_box = torch.zeros(batch, 4, grid_h, grid_w, device=boxes.device)
    scale = float(image_size)
    cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5 / scale).clamp(0, 0.999999)
    cy = ((boxes[:, 1] + boxes[:, 3]) * 0.5 / scale).clamp(0, 0.999999)
    bw = ((boxes[:, 2] - boxes[:, 0]) / scale).clamp(1e-4, 1.0)
    bh = ((boxes[:, 3] - boxes[:, 1]) / scale).clamp(1e-4, 1.0)
    ix = (cx * grid_w).long()
    iy = (cy * grid_h).long()
    for i in range(batch):
        objectness[i, 0, iy[i], ix[i]] = 1.0
        target_box[i, :, iy[i], ix[i]] = torch.stack((cx[i], cy[i], bw[i], bh[i]))
    return objectness, target_box


def gaussian_wasserstein_distance(pred: Tensor, target: Tensor) -> Tensor:
    """NWD-style distance for center-width-height boxes in normalized coordinates."""
    px, py, pw, ph = pred.unbind(dim=1)
    tx, ty, tw, th = target.unbind(dim=1)
    center = (px - tx).square() + (py - ty).square()
    shape = (torch.sqrt(pw.clamp_min(1e-6)) - torch.sqrt(tw.clamp_min(1e-6))).square()
    shape = shape + (torch.sqrt(ph.clamp_min(1e-6)) - torch.sqrt(th.clamp_min(1e-6))).square()
    return 1.0 - torch.exp(-(center + shape).sqrt() / math.sqrt(2.0))


def detector_loss(
    objectness_logits: Tensor, box_raw: Tensor, boxes: Tensor, image_size: int
) -> Tuple[Tensor, Dict[str, float]]:
    grid_h, grid_w = objectness_logits.shape[-2:]
    objectness, target_box = make_targets(boxes, grid_h, grid_w, image_size)
    pos = objectness.bool().expand_as(box_raw)
    cx = box_raw[:, 0].sigmoid()
    cy = box_raw[:, 1].sigmoid()
    wh = box_raw[:, 2:4].sigmoid()
    predicted = torch.cat((cx[:, None], cy[:, None], wh), dim=1)
    positive_pred = predicted[pos].reshape(-1, 4)
    positive_target = target_box[pos].reshape(-1, 4)
    nwd = gaussian_wasserstein_distance(positive_pred, positive_target).mean()
    bce = F.binary_cross_entropy_with_logits(objectness_logits, objectness)
    loss = bce + 5.0 * nwd
    return loss, {"loss": float(loss.detach()), "bce": float(bce.detach()), "nwd": float(nwd.detach())}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    image_size: int,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {"loss": 0.0, "bce": 0.0, "nwd": 0.0}
    count = 0
    for query, search, boxes in loader:
        query, search, boxes = query.to(device), search.to(device), boxes.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            objectness, box = model(query, search)
            loss, metrics = detector_loss(objectness, box, boxes, image_size)
        if training:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        for key in totals:
            totals[key] += metrics[key] * query.shape[0]
        count += query.shape[0]
    return {key: value / max(count, 1) for key, value in totals.items()}


def load_records(data_root: Path) -> Tuple[List[BoxRecord], List[str]]:
    annotation_path = data_root / "train" / "annotations" / "annotations.json"
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records: List[BoxRecord] = []
    video_ids: List[str] = []
    for video in payload:
        video_id = str(video["video_id"])
        if video_id not in video_ids:
            video_ids.append(video_id)
        for item in video["annotations"]:
            for raw in item.get("bboxes", []):
                box = (float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"]))
                if box[2] > box[0] and box[3] > box[1]:
                    records.append(BoxRecord(video_id, int(raw["frame"]), box))
    if not records:
        raise ValueError("No valid bounding boxes found in " + str(annotation_path))
    return records, video_ids


def export_onnx(model: nn.Module, path: Path, image_size: int, device: torch.device) -> None:
    model.eval()
    query = torch.zeros(1, 3, image_size, image_size, device=device)
    search = torch.zeros_like(query)
    torch.onnx.export(
        model,
        (query, search),
        str(path),
        input_names=["query", "search"],
        output_names=["objectness", "box"],
        opset_version=13,
        dynamic_axes={"query": {0: "batch"}, "search": {0: "batch"}},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("AeroEyes_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/aeroeyes"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.image_size < 64 or not 0.0 < args.val_fraction < 1.0:
        raise ValueError("epochs/image-size must be positive and val-fraction must be between 0 and 1")
    seed_everything(args.seed)
    data_root = args.data_root.resolve()
    records, video_ids = load_records(data_root)
    random.Random(args.seed).shuffle(video_ids)
    val_count = max(1, int(round(len(video_ids) * args.val_fraction)))
    val_ids, train_ids = video_ids[:val_count], video_ids[val_count:]
    train_set = AeroEyesDataset(data_root, records, train_ids, args.image_size, augment=True)
    val_set = AeroEyesDataset(data_root, records, val_ids, args.image_size, augment=False)
    if args.max_train_samples > 0:
        train_set.records = train_set.records[:args.max_train_samples]
    if args.max_val_samples > 0:
        val_set.records = val_set.records[:args.max_val_samples]
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    model = SiameseModulatedDetector().to(device)
    params = sum(p.numel() for p in model.parameters())
    if params > 50_000_000:
        raise RuntimeError("Model has %d parameters; Jetson limit is 50 million" % params)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    print("device=%s parameters=%d train_videos=%s val_videos=%s train_boxes=%d val_boxes=%d" %
          (device, params, ",".join(train_ids), ",".join(val_ids), len(train_set), len(val_set)))
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.image_size, scaler)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device, args.image_size, None)
        scheduler.step()
        print("epoch=%03d train_loss=%.5f val_loss=%.5f train_nwd=%.5f val_nwd=%.5f time=%.1fs" %
              (epoch, train_metrics["loss"], val_metrics["loss"], train_metrics["nwd"],
               val_metrics["nwd"], time.time() - started))
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "val_loss": val_metrics["loss"],
                      "image_size": args.image_size, "train_ids": train_ids, "val_ids": val_ids}
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
    if args.export_onnx:
        best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
        model.load_state_dict(best_checkpoint["model"])
        export_onnx(model, args.output_dir / "aeroeyes_detector.onnx", args.image_size, device)
        print("exported=%s" % (args.output_dir / "aeroeyes_detector.onnx"))


if __name__ == "__main__":
    main()
