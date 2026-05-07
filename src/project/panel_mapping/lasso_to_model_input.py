#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from model_test import PatternRetargetNet
from export_predicted_spec import replace_vertices_in_spec
from spec_to_svg import build_svg


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def measurements_to_vec(meas, body_keys):
    body = meas.get("body", meas)
    vals = []
    for k in body_keys:
        if k not in body:
            raise KeyError(f"Missing measurement: {k}")
        vals.append(float(body[k]))
    return np.asarray(vals, dtype=np.float32)


def extract_lasso_panels(lasso_json):
    """
    Extract high-resolution panel boundaries from lasso JSON.

    Priority:
      1. edge polyline geometry
      2. points
      3. vertices
      4. polygon
    """

    def panel_boundary_points(panel):
        # Best case: reconstruct from detailed edge polylines.
        if "edges" in panel and panel["edges"]:
            pts = []

            for edge in panel["edges"]:
                poly = edge.get("polyline", [])

                if not poly:
                    continue

                # Avoid duplicating shared vertices between edges.
                if not pts:
                    pts.extend(poly)
                else:
                    pts.extend(poly[1:])

            if len(pts) >= 3:
                return pts

        # Fallbacks
        return (
            panel.get("points")
            or panel.get("vertices")
            or panel.get("polygon")
        )

    panels = {}

    for p in lasso_json["panels"]:
        pid = str(p.get("id", p.get("name")))

        pts = panel_boundary_points(p)

        if pts is None:
            raise ValueError(f"Panel {pid} has no usable geometry")

        panels[pid] = np.asarray(pts, dtype=np.float32)

    return panels


def normalize_source_tensor(src_y, panel_mask):
    valid = panel_mask[:, None].astype(bool)
    pts = src_y[valid.repeat(src_y.shape[1], axis=1)].reshape(-1, 2)

    if len(pts) == 0:
        center = np.zeros(2, dtype=np.float32)
        scale = np.float32(1.0)
    else:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = ((mins + maxs) / 2.0).astype(np.float32)
        scale = np.float32(max(float((maxs - mins).max()), 1e-6))

    return ((src_y - center) / scale).astype(np.float32), center, scale


def build_source_spec_from_lasso(lasso_panels, panel_map):
    spec = {"pattern": {"panels": {}}}

    for lasso_id, gc_name in panel_map.items():
        if lasso_id not in lasso_panels:
            continue

        pts = lasso_panels[lasso_id]
        verts = [[float(x), float(y)] for x, y in pts]
        edges = [{"endpoints": [i, (i + 1) % len(verts)]} for i in range(len(verts))]

        spec["pattern"]["panels"][gc_name] = {
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "vertices": verts,
            "edges": edges,
            "label": "body"
        }

    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lasso_json", required=True)
    ap.add_argument("--panel_map", required=True)
    ap.add_argument("--source_measurements", required=True)
    ap.add_argument("--target_measurements", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    body_keys = ckpt["body_keys"]
    panel_order = ckpt["panel_order"]
    k_points = int(ckpt["k_points"])

    lasso_json = load_json(args.lasso_json)
    panel_map = load_json(args.panel_map)

    src_meas = load_yaml(args.source_measurements)
    tgt_meas = load_yaml(args.target_measurements)

    lasso_panels = extract_lasso_panels(lasso_json)

    src_y = np.zeros((len(panel_order), k_points, 2), dtype=np.float32)
    panel_mask = np.zeros((len(panel_order),), dtype=np.float32)

    for lasso_id, gc_name in panel_map.items():
        if gc_name not in panel_order:
            print(f"WARNING: {gc_name} not in checkpoint panel_order; skipping")
            continue
        if lasso_id not in lasso_panels:
            print(f"WARNING: {lasso_id} not in lasso_json; skipping")
            continue

        i = panel_order.index(gc_name)
        src_y[i] = resample_closed_polygon(lasso_panels[lasso_id], k=k_points)
        panel_mask[i] = 1.0

    src_y_norm, center, scale = normalize_source_tensor(src_y, panel_mask)

    src_body = measurements_to_vec(src_meas, body_keys)
    tgt_body = measurements_to_vec(tgt_meas, body_keys)

    if "body_mean" in ckpt and "body_std" in ckpt:
        body_mean = ckpt["body_mean"]
        body_std = ckpt["body_std"]
    else:
        print("WARNING: checkpoint has no body_mean/body_std; using body_stats_gc.json.")
        stats_path = Path(__file__).resolve().parent / "body_stats_gc.json"

        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        if stats["body_keys"] != body_keys:
            raise ValueError("body_stats_gc.json body_keys do not match checkpoint body_keys")

        body_mean = np.asarray(stats["body_mean"], dtype=np.float32)
        body_std = np.asarray(stats["body_std"], dtype=np.float32)

    src_body_norm = (src_body - body_mean) / body_std
    tgt_body_norm = (tgt_body - body_mean) / body_std

    model = PatternRetargetNet(
        body_dim=len(body_keys),
        num_panels=len(panel_order),
        k_points=k_points,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pred = model(
            torch.tensor(src_body_norm[None], dtype=torch.float32).to(device),
            torch.tensor(tgt_body_norm[None], dtype=torch.float32).to(device),
            torch.tensor(src_y_norm[None], dtype=torch.float32).to(device),
            torch.tensor(panel_mask[None], dtype=torch.float32).to(device),
        )

    pred_y_norm = pred[0].cpu().numpy()
    pred_y_abs = pred_y_norm * scale + center
    alpha = 0.35
    src_y_abs = src_y_norm * scale + center
    pred_y_abs = (1 - alpha) * src_y_abs + alpha * pred_y_abs

    source_spec = build_source_spec_from_lasso(lasso_panels, panel_map)
    pred_spec = replace_vertices_in_spec(
        source_spec,
        pred_y_abs,
        panel_order,
        panel_mask,
    )

    with open(out_dir / "source_from_lasso_specification.json", "w", encoding="utf-8") as f:
        json.dump(source_spec, f, indent=2)

    with open(out_dir / "predicted_specification.json", "w", encoding="utf-8") as f:
        json.dump(pred_spec, f, indent=2)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "body_keys": body_keys,
            "panel_order": panel_order,
            "k_points": k_points,
            "mapped_panels": panel_map,
            "scale": float(scale),
            "center": [float(center[0]), float(center[1])]
        }, f, indent=2)

    # SVG outputs
    with open(out_dir / "source_from_lasso.svg", "w", encoding="utf-8") as f:
        f.write(build_svg(source_spec))

    with open(out_dir / "predicted_pattern.svg", "w", encoding="utf-8") as f:
        f.write(build_svg(pred_spec))

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
