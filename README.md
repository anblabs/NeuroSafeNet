# NeuroSafeNet

> **Conserved biological response states organize nanoparticle biosafety across biological systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20CatBoost-orange)](https://github.com/)

---

## 📖 Overview

Predicting the biological safety and therapeutic recovery outcomes of engineered nanomaterials remains a major challenge in nanomedicine and regulatory toxicology due to physicochemical diversity, experimental heterogeneity, and context-dependent nano–bio interactions.

**NeuroSafeNet** implements an integrative, hierarchical machine-learning and data curation framework designed to decode and predict nanoparticle **biosafety** and **recovery** across heterogeneous biological systems (spanning *in vitro* and *in vivo* neurological models, and cross-organ liver and lung systems).

Through the synthesis of **599 original reports encompassing 893 experimental contexts**, this framework introduces the **Conserved Response Axis (CRA)**—a unified, mechanistically interpretable biological space organized by three foundational programmes:
1. **Pro-inflammatory signaling** (e.g., TNF-α, IL-1β, IL-6, NF-κB, p38 MAPK)
2. **Anti-inflammatory & resolution pathways** (e.g., IL-10, TGF-β, SOCS family)
3. **Apoptosis-associated responses** (e.g., Caspase-3/8/9, Cyt-c, BAX/BCL-2)

```
                       Hierarchical Feature Integration
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│        Step 1         │   │       Step 1+2        │   │      Step 1+2+3       │
│  Material Properties  │──▶│ + Exposure & Context  │──▶│ + Biological Response │
│  (Size, Shape, Zeta,  │   │  (Dose, Route, Cell,  │   │  (Pro-inf, Anti-inf,  │
│      Composition)     │   │     Injury Model)     │   │    Apoptosis / CRA)   │
└───────────────────────┘   └───────────────────────┘   └───────────────────────┘
                                                                    │
                                                                    ▼
                                                    Conserved Response Axis (CRA)
                                                    • Biosafety Prediction (~90% Bal. Acc.)
                                                    • Recovery Prediction  (~78% Bal. Acc.)
```

---

## 🔬 Manuscript Summary & Scientific Highlights

### Title
**Conserved biological response states organize nanoparticle biosafety across biological systems**

### Authors
Sri Renukadevi Balusamy<sup>1#*</sup>, Davoodbasha Mubarak Ali<sup>2,3#</sup>, Irfan Ullah<sup>4</sup>, Girish Birappa<sup>5</sup>, Samad Abdus<sup>6</sup>, Priyanka Singh<sup>7</sup>, Shadi Rahimi<sup>8</sup>, Sumit Kumar<sup>9</sup>, Mamata Karmacharya<sup>9</sup>, Yeon Ju Kim<sup>6</sup>, Suresh Ramakrishna<sup>5,10</sup>, Ivan Mijakovic<sup>8,11</sup>, Seungae Lee<sup>12*</sup>, Haribalan Perumalsamy<sup>13,14#*</sup>

<sub><sup>#</sup>These authors contributed equally | <sup>*</sup>Corresponding authors: Haribalan Perumalsamy (`harijai2004@hanyang.ac.kr`), Sri Renukadevi Balusamy (`renubalu@sejong.ac.kr`), Seungae Lee (`moon11311@naver.com`)</sub>

### Affiliations
1. Department of Food Science and Biotechnology, Sejong University, Seoul, Republic of Korea
2. School of Life Sciences, B.S. Abdur Rahman Crescent Institute of Science and Technology, Chennai, India
3. Centre Surface Technology and Applications, Korea Aerospace University, Goyang-si, Republic of Korea
4. Department of Computer Science and Engineering, Kyung Hee University (Global Campus), Yongin, South Korea
5. Graduate School of Biomedical Science and Engineering, Hanyang University, Seoul, Republic of Korea
6. Graduate School of Biotechnology, College of Life Science, Kyung Hee University, Yongin-Si, Republic of Korea
7. Department of Health Technology, Technical University of Denmark, Kongens Lyngby, Denmark
8. Division of Systems and Synthetic Biology, Chalmers University of Technology, Gothenburg, Sweden
9. Department of Biomedical Engineering, Ulsan National Institute of Science and Technology (UNIST), Ulsan, South Korea
10. College of Medicine, Hanyang University, Seoul, South Korea
11. The Novo Nordisk Foundation Biotechnology Research Institute for the Green Transition, Technical University of Denmark, Kongens Lyngby, Denmark
12. Department of Applied Chemistry and Institute of Natural Sciences, Kyung Hee University, Yongin-si, Republic of Korea
13. Center for Creative Convergence Education, Hanyang University, Seoul, Republic of Korea
14. Research Institute for Convergence of Basic Science, Hanyang University, Seoul, Republic of Korea

### Key Discoveries & Validation
- **Predictive Generalization via Hierarchical Integration**: Models incorporating biological response programmes (Step 1+2+3) significantly outperform purely physicochemical models (Step 1; Balanced Accuracy rose from 66% to 90% for biosafety, and 64% to 78% for recovery).
- **SHAP Feature Attribution**: Tree-based SHAP and ensemble interpretability identify pro-inflammatory and apoptosis-associated programmes as the dominant predictive drivers of biosafety and recovery classification.
- **Orthogonal Experimental Validation**: Evaluated using solution plasma-synthesized cellulose–cobalt nanocomposites ($C_xCo_y$) in IFN-$\gamma$-induced BV2 microglial neuroinflammation, measuring cell viability (MTT), nitric oxide (Griess), NF-$\kappa$B p65 / NOTCH2 translocation, ROS generation, and Annexin V/PI flow cytometry.
- **Functional Uptake Regulation**: Endocytic inhibition (Chlorpromazine & Methyl-$\beta$-cyclodextrin) coupled with quantitative ICP–MS confirmed that intracellular nanoparticle internalization directly drives state transitions along the CRA.
- **Cross-Scale & Cross-Organ Generalization**: Conserved response patterns validated across 3D biomimetic microglial spheroids, neurological disease transcriptomics (Alzheimer's disease [GSE142066], Multiple Sclerosis [GSE244966], Traumatic Brain Injury [GSE306006]), and cross-organ systems (Liver: GSE43515, GSE55286; Lung: GSE259384, GSE280933) across diverse nanomaterial classes (Carbon black, MWCNTs, AgNPs, Lipid Nanoparticles).

---

## 📁 Repository Structure

```
NeuroSafeNet/
├── configs/                       # Hydra / OmegaConf configuration files
│   ├── config.yaml                # Master training & cross-validation configuration
│   ├── data/                      # Dataset specifications (input/output columns, sheet names)
│   ├── io/                        # Output directories & serialization parameters
│   ├── preprocessing/             # Feature imputation & scaling parameters
│   ├── split/                     # Train/validation/test stratified splitting configurations
│   ├── paths.py                   # Global path mappings
│   └── settings.py                # Environment & pipeline settings
├── Data/                          # Curated experimental datasets
│   └── Final_Irfan_20260115.xlsx  # Harmonized nanoparticle dataset (893 contexts, 599 studies)
├── outputs/                       # Preprocessed matrices, artifacts & trained models
├── src/
│   ├── ninflam/                   # Core Python package
│   │   ├── cli.py                 # CLI entrypoint (ninflam-prep)
│   │   ├── pre_pipeline.py        # End-to-end data preprocessing pipeline
│   │   ├── config/                # Dataclass configurations
│   │   ├── core/                  # Core logging & exception handling
│   │   ├── data/                  # Loaders, cleaning, validation & normalization
│   │   ├── io/                    # Serializers & artifact managers
│   │   ├── pipeline/              # Model training & CV orchestration
│   │   ├── pre_train/             # Transformers, imputers & feature encoders
│   │   └── splitting/             # Stratified K-Fold & study-level splitting
│   ├── evaluation/                # Model evaluation metrics (AUROC, AUPRC, Balanced Acc)
│   ├── feat/                      # Feature extraction & engineering routines
│   └── models/                    # Model wrappers & definitions
├── main_preprocess.py             # Preprocessing execution script
├── main_train.py                  # Model training & evaluation script
├── pyproject.toml                 # Package dependencies & metadata
└── README.md                      # Project documentation
```

---

## 🚀 Installation & Environment Setup

### Prerequisites
- Python $\ge$ 3.9
- Recommended: Virtual environment (`venv` or `conda`)

### 1. Clone the Repository
```bash
git clone https://github.com/anblabs/NeuroSafeNet.git
cd NeuroSafeNet
```

### 2. Create and Activate Virtual Environment
```bash
# Using python venv
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

### 3. Install Package & Dependencies
Install in editable development mode:
```bash
pip install -e .
```

*Optional high-performance gradient boosting libraries:*
```bash
pip install xgboost catboost
```

---

## ⚙️ Configuration

The pipeline uses modular YAML configuration files located in `configs/`:

- **`configs/config.yaml`**: Core training models and cross-validation parameters:
  ```yaml
  log_level: INFO
  training:
    use_kfold: true
    n_splits: 10
    models:
      - RF
      - DecisionTree
      - NaiveBayes
      - LogisticRegression
      - SVM
      - KNN
      - LDA
      - MLP
      - XGBoost
    min_class_count: 2
  ```
- **`configs/data/data.yaml`**: Input feature sets (Material Properties, Exposure, CRA programmes) and target selection (`AO_Biosafety` or `AO_Recovery`).
- **`configs/split/split.yaml`**: Train (80%) / Test (20%) stratified partitioning rules.
- **`configs/preprocessing/preprocessing.yaml`**: Scaler (`standard`, `minmax`, `robust`) and categorical one-hot encoding configuration.

---

## 🛠️ Usage & Workflow

### Step 1: Data Preprocessing
Cleans raw literature-curated matrices, handles missing data via stratified imputers, encodes categorical descriptors, executes train/internal/external stratified splits, and exports serializable feature artifacts.

Run via script:
```bash
python main_preprocess.py
```
Or via CLI:
```bash
ninflam-prep --mode preprocess
```

**Generated Preprocessing Artifacts (`outputs/preprocessed_.../`):**
- `splits.npz`: Feature matrices (`X_tr`, `X_int`, `X_ext`, `y_tr`, `y_int`, `y_ext`)
- `feature_names.npy` & `feature_names.csv`: Vector of transformed feature identifiers
- `preprocessor.joblib` & `label_encoder.joblib`: Serialized transformation pipelines
- `preprocess_report.json`: Summary of feature dimensions, class counts, and split distributions

### Step 2: Model Training & Evaluation
Trains the benchmarked classifiers, executes stratified 10-fold cross-validation, saves fitted models, and computes classification metrics (Accuracy, Balanced Accuracy, Macro F1, Precision, Recall, Confusion Matrices, Prediction Probabilities).

Run via script:
```bash
python main_train.py
```
Or via CLI:
```bash
ninflam-prep --mode train
```

**Generated Training Artifacts (`outputs/training_.../`):**
- `models/*.joblib`: Serialized classifier binaries (e.g., `RF.joblib`, `SVM.joblib`)
- `metrics/*.json`: Internal and external test-set performance metrics
- `predictions/*.csv`: Ground-truth labels and prediction probabilities per class
- `confusion_matrices/*.csv`: Formatted confusion matrices
- `cross_validation/*_cv.json`: Fold-by-fold cross-validation metrics and variance summaries
- `run_metadata.json`: Comprehensive run metadata and environment specifications

---

## 📊 Supported Machine Learning Models

| Model Identifier | Algorithm Description | Implementation Backend |
| :--- | :--- | :--- |
| **`RF`** | Random Forest Classifier (`n_estimators=300`) | `scikit-learn` |
| **`XGBoost`** | Extreme Gradient Boosting Classifier | `xgboost` |
| **`CatBoost`** | Categorical Boosting Classifier | `catboost` |
| **`SVM`** | Support Vector Classifier (RBF Kernel, probability-calibrated) | `scikit-learn` |
| **`MLP`** | Multi-Layer Perceptron Neural Network (`128 x 64` hidden layers) | `scikit-learn` |
| **`LogisticRegression`** | L2-regularized Logistic Regression (`lbfgs` solver) | `scikit-learn` |
| **`DecisionTree`** | CART Decision Tree Classifier | `scikit-learn` |
| **`KNN`** | K-Nearest Neighbors Classifier (`k=5`) | `scikit-learn` |
| **`LDA`** | Linear Discriminant Analysis | `scikit-learn` |
| **`NaiveBayes`** | Gaussian Naive Bayes Classifier | `scikit-learn` |

---

## 📜 Citation

If you use NeuroSafeNet, the Conserved Response Axis (CRA) framework, or the curated dataset in your research, please cite:

```bibtex
@article{balusamy2026conserved,
  title={Conserved biological response states organize nanoparticle biosafety across biological systems},
  author={Balusamy, Sri Renukadevi and Ali, Davoodbasha Mubarak and Ullah, Irfan and Birappa, Girish and Abdus, Samad and Singh, Priyanka and Rahimi, Shadi and Kumar, Sumit and Karmacharya, Mamata and Kim, Yeon Ju and Ramakrishna, Suresh and Mijakovic, Ivan and Lee, Seungae and Perumalsamy, Haribalan},
  journal={Preprint / Under Review},
  year={2026}
}
```

---

##  Contact

For technical questions regarding the code, data pipelines, or models, please open an issue on GitHub or contact:
- **Haribalan Perumalsamy**: `harijai2004@hanyang.ac.kr`
- **Sri Renukadevi Balusamy**: `renubalu@sejong.ac.kr`
- **Seungae Lee**: `moon11311@naver.com`
