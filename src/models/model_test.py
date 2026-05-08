import argparse
import json
import tarfile
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader, random_split


def read_text_from_tar(tf: tarfile.TarFile, member_name: str) -> str:
    f = tf.extractfile(member_name)
    if f is None:
        raise FileNotFoundError(member_name)
    return f.read().decode("utf-8")


def list_element_ids(tar_path: str):
    ids = set()
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            if name.count("/") < 2:
                continue
            parts = name.split("/")
            if len(parts) >= 3:
                el = parts[1]
                if el:
                    ids.add(el)
    return sorted(ids)


def infer_body_keys(default_tar: str, random_tar: str, sample_ids):
    keys = None
    for tar_path in [default_tar, random_tar]:
        with tarfile.open(tar_path, "r:gz") as tf:
            for el in sample_ids[:50]:
                body_name = f"./{el}/{el}_body_measurements.yaml"
                try:
                    data = yaml.safe_load(read_text_from_tar(tf, body_name))
                except Exception:
                    continue
                body = data["body"]
                curr = {
                    k for k, v in body.items()
                    if not k.startswith("_") and isinstance(v, (int, float))
                }
                keys = curr if keys is None else (keys & curr)
    return sorted(keys)


def body_to_vec(body_yaml_text: str, body_keys):
    body = yaml.safe_load(body_yaml_text)["body"]
    return np.array([float(body[k]) for k in body_keys], dtype=np.float32)


def parse_design_meta(design_yaml_text: str):
    d = yaml.safe_load(design_yaml_text)["design"]
    meta = d.get("meta", {})
    upper = meta.get("upper", {}).get("v", None)
    bottom = meta.get("bottom", {}).get("v", None)
    return upper, bottom


def is_shirt_example(design_yaml_text: str) -> bool:
    upper, bottom = parse_design_meta(design_yaml_text)
    return upper == "Shirt" and bottom is None


def infer_panel_order(default_tar: str, sample_ids, filter_mode="shirt", limit=300):
    panel_names = set()
    with tarfile.open(default_tar, "r:gz") as tf:
        kept = 0
        for el in sample_ids:
            try:
                design_txt = read_text_from_tar(tf, f"./{el}/{el}_design_params.yaml")
                if filter_mode == "shirt" and not is_shirt_example(design_txt):
                    continue

                spec = json.loads(read_text_from_tar(tf, f"./{el}/{el}_specification.json"))
            except Exception:
                continue

            panel_names.update(spec["pattern"]["panels"].keys())
            kept += 1
            if kept >= limit:
                break
    return sorted(panel_names)


def resample_closed_polygon(points, k=64):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        return np.zeros((k, 2), dtype=np.float32)

    loop = np.vstack([pts, pts[:1]])
    seg = loop[1:] - loop[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())

    if total < 1e-8:
        return np.repeat(pts[:1], k, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.linspace(0.0, total, k, endpoint=False)

    out = []
    j = 0
    for t in targets:
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        a = loop[j]
        b = loop[j + 1]
        alpha = (t - cum[j]) / max(cum[j + 1] - cum[j], 1e-8)
        out.append((1 - alpha) * a + alpha * b)

    return np.asarray(out, dtype=np.float32)


def spec_to_tensor(spec_text: str, panel_order, k=64):
    spec = json.loads(spec_text)
    panels = spec["pattern"]["panels"]

    y = np.zeros((len(panel_order), k, 2), dtype=np.float32)
    mask = np.zeros((len(panel_order),), dtype=np.float32)

    for i, name in enumerate(panel_order):
        if name not in panels:
            continue
        y[i] = resample_closed_polygon(panels[name]["vertices"], k=k)
        mask[i] = 1.0

    return y, mask


def normalize_panel_tensor(src_y, tgt_y, mask):
    valid = mask[:, None].astype(bool)
    pts = src_y[valid.repeat(src_y.shape[1], axis=1)].reshape(-1, 2)

    if len(pts) == 0:
        center = np.zeros(2)
        scale = 1.0
    else:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2
        scale = max((maxs - mins).max(), 1e-6)

    return (src_y - center) / scale, (tgt_y - center) / scale


class GarmentPairDataset(Dataset):
    def __init__(self, batch_dir, k_points=64, limit=None, filter_mode="shirt"):
        self.batch_dir = Path(batch_dir)
        self.default_tar = str(self.batch_dir / "default_body" / "data.tar.gz")
        self.random_tar = str(self.batch_dir / "random_body" / "data.tar.gz")
        self.k_points = k_points
        self.filter_mode = filter_mode

        ids = sorted(set(list_element_ids(self.default_tar)) &
                     set(list_element_ids(self.random_tar)))

        filtered = []
        with tarfile.open(self.default_tar, "r:gz") as tf:
            for el in ids:
                try:
                    txt = read_text_from_tar(tf, f"./{el}/{el}_design_params.yaml")
                    if filter_mode == "shirt" and not is_shirt_example(txt):
                        continue
                    filtered.append(el)
                except:
                    continue

        if limit:
            filtered = filtered[:limit]

        self.element_ids = filtered
        self.body_keys = infer_body_keys(self.default_tar, self.random_tar, filtered)
        self.panel_order = infer_panel_order(self.default_tar, filtered)

        self.examples = []

        with tarfile.open(self.default_tar, "r:gz") as tf_def, \
             tarfile.open(self.random_tar, "r:gz") as tf_rand:

            for el in filtered:
                try:
                    sb = read_text_from_tar(tf_def, f"./{el}/{el}_body_measurements.yaml")
                    tb = read_text_from_tar(tf_rand, f"./{el}/{el}_body_measurements.yaml")
                    ss = read_text_from_tar(tf_def, f"./{el}/{el}_specification.json")
                    ts = read_text_from_tar(tf_rand, f"./{el}/{el}_specification.json")
                except:
                    continue

                src_body = body_to_vec(sb, self.body_keys)
                tgt_body = body_to_vec(tb, self.body_keys)

                src_y, sm = spec_to_tensor(ss, self.panel_order, k_points)
                tgt_y, tm = spec_to_tensor(ts, self.panel_order, k_points)

                mask = np.minimum(sm, tm)
                src_y, tgt_y = normalize_panel_tensor(src_y, tgt_y, mask)

                if mask.sum() == 0:
                    continue

                self.examples.append(dict(
                    src_body=src_body,
                    tgt_body=tgt_body,
                    src_y=src_y,
                    tgt_y=tgt_y,
                    panel_mask=mask
                ))

        bodies = np.concatenate([
            np.stack([e["src_body"] for e in self.examples]),
            np.stack([e["tgt_body"] for e in self.examples])
        ])
        self.body_mean = bodies.mean(0)
        self.body_std = bodies.std(0) + 1e-6

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        return {
            "src_body": torch.tensor((e["src_body"] - self.body_mean) / self.body_std),
            "tgt_body": torch.tensor((e["tgt_body"] - self.body_mean) / self.body_std),
            "src_y": torch.tensor(e["src_y"]),
            "tgt_y": torch.tensor(e["tgt_y"]),
            "panel_mask": torch.tensor(e["panel_mask"]),
        }


class PatternRetargetNet(nn.Module):
    def __init__(self, body_dim, num_panels, k_points):
        super().__init__()
        dim = num_panels * k_points * 2 + num_panels + 2 * body_dim
        self.num_panels = num_panels
        self.k_points = k_points

        self.net = nn.Sequential(
            nn.Linear(dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, num_panels * k_points * 2)
        )

    def forward(self, sb, tb, sy, mask):
        b = sy.shape[0]
        x = torch.cat([sy.reshape(b, -1), mask, sb, tb], dim=1)
        delta = self.net(x).view(b, self.num_panels, self.k_points, 2)
        return sy + delta * mask[:, :, None, None]


def loss_fn(pred, tgt, mask):
    return F.smooth_l1_loss(pred * mask[:, :, None, None],
                           tgt * mask[:, :, None, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--k_points", type=int, default=64)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--filter_mode", default="shirt")
    ap.add_argument("--checkpoint", default="pattern_retarget_shirt.pt")
    args = ap.parse_args()

    ds = GarmentPairDataset(args.batch_dir, args.k_points, args.limit, args.filter_mode)

    print(f"examples={len(ds)} panels={len(ds.panel_order)}")

    train_ds, val_ds = random_split(ds, [int(0.9 * len(ds)), len(ds) - int(0.9 * len(ds))])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PatternRetargetNet(len(ds.body_keys), len(ds.panel_order), args.k_points).to(device)

    if os.path.exists(args.checkpoint):
        print("Loading checkpoint...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best = float("inf")

    for e in range(args.epochs):
        model.train()
        for batch in train_loader:
            pred = model(
                batch["src_body"].to(device),
                batch["tgt_body"].to(device),
                batch["src_y"].to(device),
                batch["panel_mask"].to(device)
            )
            loss = loss_fn(pred, batch["tgt_y"].to(device), batch["panel_mask"].to(device))

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0
        for batch in val_loader:
            pred = model(
                batch["src_body"].to(device),
                batch["tgt_body"].to(device),
                batch["src_y"].to(device),
                batch["panel_mask"].to(device)
            )
            val_loss += loss_fn(pred, batch["tgt_y"].to(device), batch["panel_mask"].to(device)).item()

        val_loss /= len(val_loader)
        print(f"epoch={e+1} val={val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "body_keys": ds.body_keys,
                "panel_order": ds.panel_order,
                "k_points": args.k_points,
                "filter_mode": args.filter_mode
            }, args.checkpoint)
            print("saved")


if __name__ == "__main__":
    main()
