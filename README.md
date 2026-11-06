# AMSR-CNN: Adaptive Multi-Scale, Robust, Explainable and Reliable CNN for Brain Tumor MRI Classification

<p align="center">

**An Advanced Deep Learning Framework for Trustworthy Brain Tumor Classification Using MRI**

**Researcher:** Sakthi Gunal S  
**Institution:** Vellore Institute of Technology (VIT), India  
**Research Domain:** Deep Learning | Medical Image Analysis | CNN | Explainable AI | Trustworthy AI

</p>

---

## 1. Project Overview

This repository presents the implementation of **AMSR-CNN (Adaptive Multi-Scale, Robust, Explainable and Reliable Convolutional Neural Network)**, an advanced deep learning framework designed for four-class brain tumor classification using Magnetic Resonance Imaging (MRI).

The framework integrates adaptive MRI enhancement, multi-scale feature extraction, tumor-aware attention, adaptive feature fusion, explainable artificial intelligence, uncertainty quantification, and reliability evaluation.

The primary objective is to investigate accurate, robust, interpretable, and reliable brain tumor classification across MRI datasets with variations in image quality and data distribution.

## 2. Research Objectives

### Objective 1: Robust and Domain-Generalized CNN-Based Classification

To develop a robust and domain-generalized CNN-based framework for accurate brain tumor classification across heterogeneous and multi-source MRI datasets.

**Implementation components:**
- MRI dataset inspection and preprocessing.
- BRISC2025 dataset analysis.
- DenseNet121-based transfer learning.
- Multi-scale feature extraction.
- Tumor-aware attention mechanism.
- Adaptive Multi-Scale Fusion (AMSF).
- Multi-seed training and comparative evaluation.
- Ablation studies and performance analysis.

### Objective 2: Adaptive Preprocessing and Feature Enhancement

To design an adaptive preprocessing and feature enhancement pipeline integrated with transfer learning, data augmentation, and hyperparameter tuning to improve model accuracy, efficiency, and generalization.

**Implementation components:**
- Data Quality Assessment (DQA).
- Adaptive MRI Enhancement Module (AMEM).
- Intensity normalization.
- Non-local means denoising.
- CLAHE-based contrast enhancement.
- Adaptive sharpening.
- Background suppression.
- Brain Region of Interest (ROI) extraction.
- Data augmentation and hyperparameter optimization.

### Objective 3: Explainable and Uncertainty-Aware Deep Learning

To develop an explainable and uncertainty-aware deep learning framework that provides transparent, reliable, and clinically interpretable predictions for brain tumor diagnosis.

**Implementation components:**
- Grad-CAM++ visualization.
- Integrated Gradients.
- Occlusion-based explanation.
- Prediction confidence estimation.
- Predictive entropy.
- Monte Carlo dropout.
- Deep ensemble uncertainty estimation.
- Temperature scaling and calibration.
- Reliability diagrams and risk-coverage analysis.

---

## 3. Proposed AMSR-CNN Architecture

The proposed framework consists of interconnected modules for MRI quality assessment, adaptive enhancement, multi-scale feature learning, classification, explainability, and reliability evaluation.

### Overall Workflow

```text
             MRI DATASET
                  |
                  v
        DATA INSPECTION
                  |
                  v
       DATA QUALITY ASSESSMENT
                 (DQA)
                  |
                  v
     ADAPTIVE MRI ENHANCEMENT
                (AMEM)
                  |
                  v
       BRAIN ROI EXTRACTION
                  |
                  v
       DenseNet121 BACKBONE
         (Transfer Learning)
                  |
        +---------+---------+
        |         |         |
        v         v         v
      Fine   Intermediate  Deep
     Features  Features   Features
        |         |         |
        +---------+---------+
                  |
                  v
       TUMOR-AWARE ATTENTION
                  |
                  v
      ADAPTIVE MULTI-SCALE
          FEATURE FUSION
                  |
                  v
       CLASSIFICATION HEAD
                  |
                  v
       FOUR-CLASS PREDICTION
                  |
        +---------+---------+
        |                   |
        v                   v
  EXPLAINABILITY        RELIABILITY
     ANALYSIS             ANALYSIS
        |                   |
        v                   v
    Grad-CAM++         Calibration
    Integrated         Uncertainty
    Gradients          Confidence
    Occlusion          Risk-Coverage
```

### Key Architectural Components

| Component | Function |
|---|---|
| DQA | Evaluates MRI image quality |
| AMEM | Performs quality-guided image enhancement |
| ROI | Extracts brain-focused image regions |
| DenseNet121 | Extracts deep hierarchical features |
| Multi-scale learning | Captures features at different resolutions |
| Tumor-aware attention | Learns relevant channel and spatial features |
| AMSF | Adaptively combines multi-scale representations |
| Classification head | Predicts four tumor categories |
| Explainability | Provides visual feature attribution |
| Uncertainty module | Evaluates prediction confidence and uncertainty |

---

## 4. Dataset Description

**Primary Dataset: BRISC2025**

- Source: Publicly available BRISC2025 dataset.
- Modality: T1-weighted Brain MRI.
- Task: Four-class brain tumor classification.
- Training samples: 5,000.
- Testing samples: 1,000.
- Total primary samples: 6,000.

### Classification Categories

1. Glioma
2. Meningioma
3. Pituitary Tumor
4. No Tumor

**External validation:** The notebook supports optional automatic detection of an independent four-class brain tumor MRI dataset attached to the execution environment.

The external dataset is reserved for zero-shot evaluation and is not used for training, model selection, calibration, or hyperparameter optimization.

**Dataset access:** [BRISC2025 on Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)

Dataset files are not distributed through this repository. Users must obtain the dataset from its original source and comply with its licensing and usage conditions.

---

## 5. Technical Methodology

### 5.1 Data Quality Assessment (DQA)

The framework evaluates MRI quality using image-derived descriptors:

- Sharpness.
- Noise estimation.
- Contrast.
- Dynamic range.
- Intensity non-uniformity.
- Saturation.
- Background residual.

A quality index is calculated to characterize the input MRI quality.

### 5.2 Adaptive MRI Enhancement Module (AMEM)

AMEM performs deterministic, quality-guided image enhancement.

The processing stages include:

1. Percentile-based intensity normalization.
2. Adaptive non-local means denoising.
3. CLAHE-based contrast enhancement.
4. Adaptive unsharp masking.
5. Background suppression.

### 5.3 Brain ROI Extraction

A foreground mask is used to identify the brain-focused region. A square bounding box with a 4% margin is extracted and resized to a consistent input resolution.

### 5.4 DenseNet121-Based Multi-Scale Feature Learning

DenseNet121 is used as the feature extraction backbone with ImageNet transfer learning.

Three feature representations are extracted:

- Fine-scale features.
- Intermediate-scale features.
- Deep-scale features.

### 5.5 Tumor-Aware Attention

Channel attention and spatial attention mechanisms are integrated to emphasize informative feature channels and spatial regions.

### 5.6 Adaptive Multi-Scale Fusion (AMSF)

The multi-scale representations are combined using learnable fusion weights:

\[
F=\alpha_1F_1+\alpha_2F_2+\alpha_3F_3
\]

where the fusion weights are learned using a softmax operation.

### 5.7 Classification

The fused representation is passed through the classification head to generate predictions for the four brain MRI categories.

---

## 6. Training Strategy

The implementation incorporates:

- AdamW optimizer.
- Discriminative learning rates.
- Linear warm-up and cosine learning-rate scheduling.
- Class-weighted cross-entropy loss.
- Mixed-precision training.
- Gradient clipping.
- GPU-based data augmentation.
- Early stopping based on validation macro-F1.
- Multiple independent random seeds.

The official BRISC test split is reserved for final evaluation rather than model selection or calibration.

---

## 7. Explainable Artificial Intelligence (XAI)

Three explanation approaches are implemented.

| Method | Purpose |
|---|---|
| Grad-CAM++ | Visualizes spatially relevant feature activations |
| Integrated Gradients | Estimates pixel-level attribution |
| Occlusion | Measures prediction changes after masking image regions |

The notebook also includes quantitative evaluation of localization and explanation faithfulness using tumor masks where available.

---

## 8. Uncertainty and Reliability Evaluation

The framework investigates prediction reliability through:

**Confidence estimation**
- Maximum softmax probability.
- Predictive entropy.

**Uncertainty estimation**
- Monte Carlo dropout.
- Deep ensemble.
- Epistemic uncertainty.
- Total predictive uncertainty.

**Calibration**
- Expected Calibration Error (ECE).
- Maximum Calibration Error (MCE).
- Brier Score.
- Negative Log-Likelihood (NLL).
- Temperature scaling.

**Selective prediction**
- Risk-coverage curves.
- Area Under the Risk-Coverage Curve (AURC).

---

## 9. Robustness Evaluation

The framework evaluates model performance under MRI-motivated image perturbations.

| Perturbation | Description |
|---|---|
| Noise | Rician noise |
| Blur | Gaussian blur |
| Contrast variation | Contrast reduction |
| Intensity variation | Smooth multiplicative bias-field variation |

Four severity levels are evaluated for each perturbation to investigate performance degradation and robustness.

---

## 10. Experimental Evaluation

The implementation includes the following evaluation procedures:

- Accuracy.
- Precision.
- Recall / Sensitivity.
- Specificity.
- Macro-F1 score.
- ROC-AUC.
- Precision-Recall curves.
- Confusion matrix.
- Bootstrap confidence intervals.
- Multi-seed performance analysis.
- Ablation experiments.
- Statistical significance testing.
- Error analysis.
- External validation.

### Ablation Study

The following model configurations are included:

1. DenseNet121.
2. DenseNet121 + AMEM.
3. MS-Fixed.
4. MS-Fixed + Attention.
5. AMSR-CNN (Full Model).
6. AMSR-CNN without AMEM.

These configurations enable investigation of the individual contributions of preprocessing, multi-scale feature learning, attention, and adaptive fusion.

### Results Reporting

Actual experimental results should be added after verifying the generated evaluation tables.

| Evaluation | Result |
|---|---|
| Test accuracy | To be populated |
| Macro-F1 | To be populated |
| ROC-AUC | To be populated |
| ECE | To be populated |
| Robustness retention | To be populated |
| External validation | To be populated |

---

## 11. Repository Structure

```text
AMSR-CNN/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── objective-01.ipynb
│
├── Objective_1/
│   ├── dataset_analysis/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── Objective_2/
│   ├── DQA/
│   ├── AMEM/
│   ├── ROI_extraction/
│   ├── augmentation/
│   └── feature_enhancement/
│
├── Objective_3/
│   ├── explainability/
│   ├── uncertainty/
│   ├── calibration/
│   └── reliability/
│
├── experiments/
│   ├── ablation/
│   ├── robustness/
│   ├── external_validation/
│   └── statistical_analysis/
│
├── results/
│   ├── tables/
│   ├── metrics/
│   └── training_logs/
│
├── figures/
│   ├── architecture/
│   ├── confusion_matrix/
│   ├── ROC_curves/
│   ├── gradcam/
│   └── reliability/
│
└── documentation/
    ├── methodology.md
    └── experimental_protocol.md
```

*The directory structure is the recommended organization for the repository. The uploaded notebook currently contains the integrated implementation; the objective-wise folders can be populated as the code is modularized.*

---

## 12. Implementation Environment

**Development Platform:** Kaggle Notebook  
**Recommended Hardware:** NVIDIA GPU (T4 / P100)  
**Programming Language:** Python  
**Deep Learning Framework:** PyTorch  
**Feature Extraction:** Torchvision DenseNet121

### Main Libraries

- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Captum (if separately used for attribution implementation)

### Execution

1. Open the notebook in Kaggle.
2. Attach the BRISC2025 dataset.
3. Enable GPU acceleration.
4. Execute the notebook sequentially.
5. Attach an independent four-class dataset for optional external validation.
6. Review generated experimental tables, figures and saved model checkpoints.

Refer to the notebook for the exact execution sequence and configuration.

---

## 13. Reproducibility and Research Integrity

The implementation incorporates:

- Fixed random seeds.
- Stratified train/validation partitioning.
- Independent test evaluation.
- Near-duplicate image auditing.
- Multi-seed performance reporting.
- Bootstrap confidence intervals.
- Ablation-based component analysis.
- Statistical validation.
- Checkpoint and result caching.
- Automated figure and table export.

**Important limitations:**

1. BRISC2025 does not provide patient identifiers. The perceptual-hash audit cannot guarantee patient-level independence between splits.
2. The extracted ROI represents the head/brain foreground region and does not constitute skull stripping.
3. Image-quality metric improvements alone do not establish diagnostic benefit.
4. The implementation focuses on 2D T1-weighted MRI slices and slice-level classification.
5. External validation and prospective clinical validation must be reported according to the experiments actually completed.

---

## 14. Research Contributions

The research investigates five interconnected dimensions:

1. **Adaptive MRI Enhancement:** Quality-guided preprocessing to address image-quality variations.
2. **Multi-Scale Feature Learning:** Hierarchical feature extraction using DenseNet121.
3. **Tumor-Aware Attention:** Spatial and channel-wise feature refinement.
4. **Explainable AI:** Visual and quantitative investigation of model explanations.
5. **Trustworthy Prediction:** Calibration, uncertainty estimation, and robustness analysis.

---

## 15. Researcher Information

**Sakthi Gunal S**  
Research Scholar  
Vellore Institute of Technology (VIT), India

**Research Interests:**
- Deep Learning
- Convolutional Neural Networks
- Medical Image Processing
- Brain Tumor Classification
- Explainable Artificial Intelligence
- Uncertainty Quantification
- Trustworthy AI

---

## 16. Disclaimer

This repository is intended exclusively for academic and research purposes.

The proposed AMSR-CNN framework is an experimental research implementation. Its predictions are not a substitute for professional medical diagnosis, radiological interpretation, or clinical decision-making.

---

## 17. Citation

If this implementation contributes to your research, please cite the associated research publication when it becomes available.

**Publication details:** To be added after publication.

---

**Developed by Sakthi Gunal S | Research Scholar | VIT, India**

⭐ If you find this research implementation useful, consider starring the repository and citing the associated publication when available.
