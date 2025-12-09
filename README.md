# Reproducing Geneformer: Transfer Learning in Network Biology
## Project Overview
This repository contains the code and analysis for reproducing Geneformer, a transformer-based deep learning model pretrained on nearly 30 million human single-cell transcriptomes. Our study benchmarks Geneformer's performance on gene and cell classification tasks against traditional machine learning methods like Support Vector Machines (SVMs), Random Forests (RFs), and XGBoost.

## Key Objectives
- **Reproduce Core Capabilities**: Validate Geneformer's efficacy in predicting cell types and gene dosage sensitivity.

- **Benchmark Performance**: Compare deep learning transfer methods against established baselines in data-limited scenarios.

- **Analyze Generalization**: Assess how rank-based encoding and pretraining affect model robustness across different biological contexts.

## Methodology
Geneformer leverages a self-supervised masked prediction objective to learn contextual gene relationships.

- **Pretraining**: Utilized Genecorpus-30M with a rank-based encoding approach to prioritize cell-state-specific regulators.

- **Data Processing**: Processed single-cell RNA sequencing data from human thymus samples (cTEC and mTEC), tokenizing them for transformer input.

- **Fine-Tuning**: The model was fine-tuned for downstream tasks using frozen layers to prevent overfitting to limited labeled data.

## Results
### 1. Cell Classification (cTEC vs. mTEC)

Geneformer demonstrated exceptional ability to distinguish between distinct cell types.

- **Accuracy**: Achieved 96.33% accuracy and a Macro F1-score of 0.96.

- **ROC AUC**: Reached 0.99, indicating excellent class separation.

- **Comparison**: Performance was comparable to optimized traditional models (e.g., XGBoost), likely due to the strong biological distinctness between cortical and medullary thymic epithelial cells.

- **Visualization**: UMAP embeddings showed clear, distinct clusters for the two cell populations.

### 2. Gene Classification (Dosage Sensitivity)

We evaluated the model's ability to predict dosage-sensitive vs. dosage-insensitive transcription factors.

- **Performance**: Achieved a Macro F1 score of 0.82 and an AUC of 0.86.

- **Class Imbalance**: While the model predicted 92% of dosage-insensitive factors accurately, it faced challenges with dosage-sensitive factors (69% accuracy) due to limited label coverage.

- **Baselines**: Geneformer generally outperformed baseline models (Logistic Regression, SVM) which required self-training/label propagation to achieve competitive results.

## Performance Comparison

We benchmarked Geneformer against traditional machine learning baselines (Logistic Regression, SVM, Random Forest, XGBoost).

### Cell Classification
Traditional models like XGBoost and Random Forest achieved comparable accuracy to Geneformer in binary cell type classification, likely due to the distinct biological separation between cTEC and mTEC populations.

![Cell Classification Benchmarks](plots/figure4_cell_performance.png)
*Figure: Comparison of accuracy and confidence intervals between Geneformer and baseline models for cell classification.*

### Gene Classification (Dosage Sensitivity)
In the more complex task of predicting dosage sensitivity, Geneformer demonstrated robust performance (AUC 0.86), outperforming several baselines that required self-training/label propagation to handle missing labels.

![Gene Classification ROC](plots/figure6_gene_roc.png)
*Figure: ROC Curves comparing Geneformer against baseline models.*

![Gene Classification F1 Scores](plots/figure7_gene_f1.png)
*Figure: F1 Scores comparing Geneformer against baseline models.*

## Tech Stack
- **Core Model**: Geneformer (Hugging Face Transformers)

- **Data Handling**: Scanpy, AnnData 

- **Machine Learning**: Scikit-learn, XGBoost 

- **Visualization**: UMAP, Matplotlib
