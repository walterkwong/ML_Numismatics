## Coin Grading

### Goal

The grading pipeline estimates a coin's Sheldon grade from 1–70 from a photograph.

### Augmentation

Grading uses more conservative augmentation to preserve surface details. This includes:

* Rotation and flipping
* Small geometric transformations
* Perspective changes
* Brightness, contrast and colour changes
* Noise
* JPEG compression

### Models

Two types of grading model are compared.

**Deep models**

The following pretrained vision backbones are compared:

| Name            | Backbone         |
| --------------- | ---------------- |
| `dinov2`        | DINOv2           |
| `ijepa`         | I-JEPA           |
| `clip`          | CLIP             |
| `convnext_tiny` | ConvNeXt V2 Tiny |

The backbones are used with frozen linear heads, LoRA adaptation or full fine-tuning where applicable.

**Handcrafted baseline**

The feature design is based on the approach described in [this paper](https://arxiv.org/pdf/2512.04464). These features are passed through a small MLP.

The models predict three related targets:

* Circulated vs uncirculated state
* Adjectival grade tier
* Numeric Sheldon grade

Two grade heads are evaluated:

* **CORAL** for ordinal grade prediction
* **Gaussian regression** for grade prediction with uncertainty

The three outputs are combined using Bayesian fusion to produce a final grade distribution.

### Evaluation

The grading pipeline measures:

* MAE in grade points
* Predictions within 1, 3 and 5 grade points
* State accuracy
* Tier accuracy
* Inference latency
* Model size

### Key Findings

DINOv2 with the Gaussian regression head achieved 88.24% state accuracy, 52.14% tier accuracy and a grade MAE of 4.51, with a mean latency of 8.77 ms.

While the system provides a useful low-latency estimate, its accuracy is insufficient for reliable mint-state coin valuation, where small differences in grade can have a substantial effect on coin price.

Machine learning provides a low-latency approach to coin grading with reasonable predictive performance, but it cannot substitute for professional grading services given the limitations of making predictions from a single coin image.
