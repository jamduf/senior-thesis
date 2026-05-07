# Trained Models

Pretrained garment retargeting checkpoints used for body-conditioned sewing pattern geometry prediction.

---

# Overview

The models in this directory predict updated garment panel geometry given:

- source garment panel geometry
- source body measurements
- target body measurements

The models operate on structured sewing pattern panel representations derived from the GarmentCode framework and the lasso extraction pipeline.

---

# Current Checkpoints

| Model | Garment Type | Status |
|---|---|---|
| `pattern_retarget_shirt_v2.pt` | Shirts / tops | Experimental |
| `pattern_retarget_pants_only_v1.pt` | Pants | Experimental |

---

# Input Representation

Each model consumes:

- semantic garment panel labels
- resampled polygon boundary points
- source body measurements
- target body measurements

Example panel labels:

```text
left_ftorso
right_ftorso
left_btorso
right_btorso
left_sleeve_f
```

---

# Measurement Format

Body measurements follow the GarmentCode measurement schema.

Example:

```yaml
body:
  bust: 89.0
  waist: 71.0
  hips: 97.0
  shoulder_w: 38.0
  arm_length: 56.0
```

Measurement templates are provided in:

```text
src/project/panel_mapping/size_charts/
```

---

# Training Data

The checkpoints were trained on processed garment geometry and body measurement data derived from the GarmentCode framework.

GarmentCode:
- https://github.com/maria-korosteleva/GarmentCode

The models were trained on procedural garment/body variations generated from GarmentCode assets.

---

# Running Inference

Example:

```bash
./run_lasso_to_model.sh \
  pattern_project.json \
  source_measurements.yaml \
  target_measurements.yaml \
  checkpoint.pt \
  output_dir
```

The inference pipeline produces:

```text
source_from_lasso.svg
predicted_pattern.svg
predicted_specification.json
```

# Current Limitations

These checkpoints are research prototypes and currently work best when:

- garment topology resembles training data
- patterns contain relatively few decorative details
- garment patterns are intended for non-stretch woven fabrics

Known limitations:

- limited training distribution
- procedural dataset domain mismatch
- incomplete support for highly detailed garments
- no seam allowance generation

---

