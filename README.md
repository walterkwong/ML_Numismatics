# Machine Learning in Numismatics: Automated Rare Coin Classification and Valuation System

Research project investigating automated coin analysis using computer vision. It contains three independent components:
1. [**Segmentation**](https://github.com/walterkwong/ML_Numismatics/tree/main/YOLO): detects and segments coins in an image using YOLO.
2. [**Classification**](https://github.com/walterkwong/ML_Numismatics/tree/main/classification): identifies a coin's variety, family, category, face value and year.
3. [**Grading**](https://github.com/walterkwong/ML_Numismatics/tree/main/grading): estimates a coin's Sheldon grade (1–70) from a photograph.

The classification and grading pipelines compare pretrained vision backbones with different adaptation methods and use multi-task prediction with Bayesian fusion.

The main contribution is an assessment of the viability of performing coin classification and grading from a single coin image for convenient use, rather than relying on traditional two-sided evaluations and coarse-grained predictions with higher latency.

A web application was created to demonstrate the system's viability.

**Repository:** [AutoCoinScan Repo](https://github.com/walterkwong/AutoCoinNet)

**Demo:** [AutoCoinScan Web App](https://autocoinnet-web.coinrecognitionval.workers.dev/)

> **Note:** The GPU backend was shut down on 28/09/2026 due to budget constraints.

## Key Findings

1. Machine learning shows potential for coin valuation, but accessible datasets remain limited and automated methods are inherently constrained compared with physical examination.
2. The system demonstrates the viability of machine learning for numismatic analysis, but further work is required to improve model accuracy and interpretability.
