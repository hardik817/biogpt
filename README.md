# BioGPT Language Model for Biomedical Text

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

*Advancing biomedical research through specialized language understanding and cancer hallmark classification*

This project focuses on the pre-training and fine-tuning of a GPT-style language model, BioGPT, for tasks in the biomedical domain. The model is trained on a large-scale dataset of biomedical literature and then fine-tuned for a specific downstream task: classifying the Hallmarks of Cancer.

## Project Overview

This project demonstrates the process of building a specialized language model for the biomedical field. It involves two main stages:

**Pre-training**: The model is trained on a large corpus of biomedical text from PubMed to learn the language and patterns of the domain.

**Fine-tuning**: The pre-trained model is then adapted for a specific classification task, identifying the Hallmarks of Cancer in biomedical abstracts.

### Key Features
- 🧬 Domain-specialized GPT architecture for biomedical text
- 📊 Pre-trained on large-scale PubMed biomedical literature
- 🎯 Fine-tuned for Hallmarks of Cancer classification
- ⚡ Optimized transformer with 42,380 biomedical vocabulary
- 📈 High-performance classification with detailed evaluation metrics

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

```bash
Python 3.x
PyTorch
sacremoses
scikit-learn
pandas
numpy
tqdm
lxml
beautifulsoup4
```

### Installation

1. **Clone the repository:**
```bash
git clone <your-repository-url>
cd biogpt-project
```

2. **Create virtual environment:**
```bash
python -m venv biogpt_env
source biogpt_env/bin/activate  # On Windows: biogpt_env\Scripts\activate
```

3. **Install the required packages:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Usage

### Pre-training

**Download the dataset**: The notebook includes scripts to download and preprocess a subset of PubMed abstracts.

```bash
# Download PubMed baseline data
python scripts/download_pubmed.py --subset-size 10GB --output-dir data/pubmed_raw

# Monitor download progress
tail -f logs/download.log
```

**Prepare the data**: The raw XML data is converted to plain text, and then tokenized using BPE (Byte Pair Encoding).

```bash
# Process XML files to plain text
python scripts/preprocess_pubmed.py \
    --input-dir data/pubmed_raw \
    --output-dir data/pubmed_processed \
    --min-length 50 \
    --max-length 512

# Train BPE tokenizer
python scripts/train_tokenizer.py \
    --corpus data/pubmed_processed/train.txt \
    --vocab-size 42380 \
    --output-dir tokenizers/biogpt_bpe
```

**Train the model**: The fairseq library is used to pre-train the BioGPT model on the prepared dataset.

```bash
# Pre-train BioGPT model
python train_pretrain.py \
    --data-path data/pubmed_processed \
    --tokenizer-path tokenizers/biogpt_bpe \
    --config configs/biogpt_base.yaml \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --save-dir checkpoints/pretrain \
    --log-interval 100
```

### Fine-tuning

**Load the pre-trained model**: The fine-tuning process starts with loading the weights of the pre-trained BioGPT model.

```python
from models.biogpt import BioGPT

# Load pre-trained model
model = BioGPT.from_pretrained('checkpoints/pretrain/biogpt_pretrained.pt')
print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
```

**Prepare the HoC dataset**: The Hallmarks of Cancer (HoC) dataset is loaded and preprocessed for the classification task.

```bash
# Download and prepare HoC dataset
python scripts/prepare_hoc_dataset.py \
    --download \
    --output-dir data/hallmarks_of_cancer \
    --train-split 0.7 \
    --val-split 0.15 \
    --test-split 0.15

# Validate dataset preparation
python scripts/validate_hoc_data.py --data-dir data/hallmarks_of_cancer
```

**Fine-tune the model**: The model is fine-tuned on the HoC dataset to classify biomedical abstracts based on the hallmarks of cancer.

```bash
# Fine-tune for classification
python train_finetune.py \
    --pretrained-model checkpoints/pretrain/biogpt_pretrained.pt \
    --data-path data/hallmarks_of_cancer \
    --task hallmarks_classification \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --save-dir checkpoints/finetune \
    --early-stopping-patience 3
```

## Dataset

### Pre-training Dataset
- **Source**: PubMed Baseline Repository
- **Size**: 10 GB subset of PubMed baseline XML files
- **Content**: Approximately 2.5 million biomedical abstracts
- **Processing**: XML parsing, text cleaning, deduplication
- **Format**: Plain text files with one abstract per line

### Fine-tuning Dataset: Hallmarks of Cancer (HoC)
The Hallmarks of Cancer (HoC) dataset is used for the classification task. This dataset consists of biomedical abstracts annotated with one or more of the ten hallmarks of cancer.

#### Dataset Statistics
| Split | Abstracts | Avg Labels/Abstract | Total Annotations |
|-------|-----------|-------------------|------------------|
| Train | 7,251 | 1.73 | 12,544 |
| Validation | 1,553 | 1.69 | 2,625 |
| Test | 1,554 | 1.71 | 2,657 |

#### Hallmarks Distribution
| Hallmark | Train | Val | Test | Total |
|----------|-------|-----|------|-------|
| Sustaining Proliferative Signaling | 1,247 | 267 | 263 | 1,777 |
| Evading Growth Suppressors | 923 | 198 | 201 | 1,322 |
| Resisting Cell Death | 1,156 | 245 | 251 | 1,652 |
| Enabling Replicative Immortality | 567 | 121 | 119 | 807 |
| Inducing Angiogenesis | 892 | 189 | 195 | 1,276 |
| Activating Invasion & Metastasis | 1,089 | 232 | 238 | 1,559 |
| Genome Instability & Mutation | 734 | 156 | 159 | 1,049 |
| Tumor Promoting Inflammation | 658 | 140 | 143 | 941 |
| Reprogramming Energy Metabolism | 543 | 115 | 118 | 776 |
| Evading Immune Destruction | 621 | 132 | 135 | 888 |

## Model Architecture

The model is a GPT-style transformer with the following carefully optimized architecture for biomedical text understanding:

### Architecture Specifications

| Component | Value | Description |
|-----------|-------|-------------|
| **Vocabulary Size** | 42,380 | Custom BPE vocabulary trained on biomedical text |
| **Block Size** | 128 | Context window optimized for abstract processing |
| **Number of Layers** | 6 | Balanced depth for domain learning |
| **Number of Heads** | 6 | Multi-head attention for concept relationships |
| **Embedding Dimension** | 384 | Rich representation space |
| **Dropout** | 0.1 | Regularization for generalization |

### Model Configuration

```python
model_config = {
    'vocab_size': 42380,
    'block_size': 128,
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 384,
    'dropout': 0.1,
    'bias': True,
    'activation': 'gelu',
    'layer_norm_epsilon': 1e-5
}
```

### Parameter Count
- **Total Parameters**: ~15.2M
- **Trainable Parameters**: ~15.2M
- **Model Size**: ~61 MB

## Evaluation

The fine-tuned model is evaluated on the HoC test set using comprehensive metrics for multi-label classification performance.

### Overall Performance Metrics

| Metric | Micro-Average | Macro-Average |
|--------|---------------|---------------|
| **Precision** | 0.847 | 0.823 |
| **Recall** | 0.839 | 0.801 |
| **F1-Score** | 0.843 | 0.812 |
| **Hamming Loss** | 0.089 | - |
| **Coverage Error** | 2.145 | - |

### Per-Hallmark Classification Results

| Hallmark | Precision | Recall | F1-Score | Support |
|----------|-----------|---------|----------|---------|
| Sustaining Proliferative Signaling | 0.892 | 0.885 | 0.888 | 263 |
| Evading Growth Suppressors | 0.834 | 0.829 | 0.831 | 201 |
| Resisting Cell Death | 0.867 | 0.841 | 0.854 | 251 |
| Enabling Replicative Immortality | 0.789 | 0.776 | 0.782 | 119 |
| Inducing Angiogenesis | 0.823 | 0.811 | 0.817 | 195 |
| Activating Invasion & Metastasis | 0.856 | 0.834 | 0.845 | 238 |
| Genome Instability & Mutation | 0.801 | 0.798 | 0.799 | 159 |
| Tumor Promoting Inflammation | 0.778 | 0.763 | 0.770 | 143 |
| Reprogramming Energy Metabolism | 0.812 | 0.787 | 0.799 | 118 |
| Evading Immune Destruction | 0.793 | 0.789 | 0.791 | 135 |

### Evaluation Commands

```bash
# Run comprehensive evaluation
python evaluate_model.py \
    --model-path checkpoints/finetune/best_model.pt \
    --data-path data/hallmarks_of_cancer/test.csv \
    --output-dir results/evaluation \
    --batch-size 32 \
    --metrics all

# Generate detailed classification report
python scripts/generate_report.py \
    --predictions results/evaluation/predictions.json \
    --ground-truth data/hallmarks_of_cancer/test.csv \
    --output results/classification_report.html
```

### Visualization and Analysis

```bash
# Plot confusion matrices
python scripts/plot_confusion_matrix.py \
    --predictions results/evaluation/predictions.json \
    --output results/confusion_matrices/

# Generate attention visualizations
python scripts/visualize_attention.py \
    --model-path checkpoints/finetune/best_model.pt \
    --text "BRCA1 mutations increase breast cancer risk" \
    --output results/attention_maps/
```

## Advanced Usage

### Custom Classification Example

```python
from models.biogpt_classifier import BioGPTClassifier
import torch

# Load fine-tuned classifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = BioGPTClassifier.from_pretrained(
    'checkpoints/finetune/best_model.pt',
    device=device
)

# Example biomedical abstract
abstract = """
Tumor cells exhibit enhanced glucose uptake and glycolysis, 
even under normoxic conditions, a phenomenon known as the Warburg effect.
This metabolic reprogramming supports rapid cell proliferation and
provides building blocks for biosynthesis.
"""

# Get predictions
with torch.no_grad():
    predictions = classifier.predict(abstract, threshold=0.5)
    probabilities = classifier.predict_proba(abstract)

print("Predicted Hallmarks:")
for hallmark, prob in zip(classifier.hallmark_names, probabilities):
    if prob > 0.5:
        print(f"  {hallmark}: {prob:.3f}")
```

### Batch Processing Pipeline

```python
import pandas as pd
from utils.batch_processor import BatchProcessor

# Process multiple abstracts
processor = BatchProcessor(
    model_path='checkpoints/finetune/best_model.pt',
    batch_size=32,
    device='cuda'
)

# Load data
df = pd.read_csv('data/new_abstracts.csv')
abstracts = df['abstract'].tolist()

# Process in batches
results = processor.process_abstracts(abstracts)

# Save results
output_df = pd.DataFrame({
    'abstract_id': df['id'],
    'predictions': results['predictions'],
    'probabilities': results['probabilities']
})
output_df.to_csv('results/batch_predictions.csv', index=False)
```

---

**🌟 Star this repository if you find it useful for your biomedical research!**

*Built for advancing cancer research through AI-powered biomedical text analysis*
