## Coin Classification

### Goal

The classification pipeline identifies a coin's exact variety from a photograph while also predicting its family, category, face value and year.

The evaluation uses varieties held out from training. Reference images from these varieties form a gallery, while augmented versions are used as queries for retrieval.

### Augmentation

Training images are augmented to make clean catalogue images more representative of real photographs. This includes:

* Background replacement
* Rotation, scaling and translation
* Perspective distortion
* Brightness and contrast changes
* Gamma and lighting effects
* Blur and noise
* Image compression

Evaluation uses only resizing and normalisation.

### Models

The following pretrained vision backbones are compared:

| Name            | Backbone         |
| --------------- | ---------------- |
| `dinov2`        | DINOv2           |
| `ijepa`         | I-JEPA           |
| `clip`          | CLIP             |
| `convnext_tiny` | ConvNeXt V2 Tiny |

The backbones are used with frozen linear heads, LoRA adaptation or full fine-tuning where applicable.

A multi-task model predicts:

* Family
* Category
* Variety
* Face value
* Year

The variety prediction is evaluated through image retrieval using cosine similarity rather than the classification head. This allows the system to identify varieties that were not present during training, given the limited availability of suitable datasets.

A hierarchical re-ranking step combines visual similarity with the predicted family, category, face value and year. This produces the final variety posterior and also refines the category, face value and year predictions.

### Evaluation

The classification pipeline measures:

* Retrieval Rank-1 and Rank-5
* Family accuracy
* Category accuracy
* Face-value accuracy
* Year MAE
* Inference latency
* Model size

### Key Findings

ConvNeXt performed best for the classification tasks, achieving 95.16% family accuracy, 92.90% category accuracy and 93.59% face-value accuracy, with a mean latency of 7.94 ms.

However, evaluation of coin varieties was affected by instance-level memorisation, limiting the ability to assess generalisation to unseen varieties reliably.

Computer vision architectures show strong potential for coin classification, but limited access to high-quality data remains a major bottleneck for robust model training and evaluation.