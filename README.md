# GlossEx: Automatic Extraction of Domain-Specific Terms

This project implements an end-to-end pipeline for **automatic extraction of domain-specific terminology** from textual data, inspired by recent research on glossary extraction using contextualized embeddings.

The implementation is **written from scratch** and adapted to a **new English-domain corpus**, as required by the course assignment.

---

## Project Overview

The goal of this project is to automatically identify **economics-related terms** from an English educational corpus using a combination of:
- statistical saliency measures
- contextualized word embeddings (BERT)
- unsupervised clustering
- weak supervision via seed lists

The overall methodology is inspired by prior work on Glossary Extraction (GlossEx), but the **entire pipeline was implemented independently** and applied to **new data collected by the author**.

---

## Pipeline Summary

The implemented pipeline consists of the following steps:

1. **Text Preprocessing**
   - Tokenization
   - Lowercasing
   - Stopword removal

2. **Saliency / Specificity Scoring**
   - Tokens are ranked based on their domain frequency compared to general-language frequency.

3. **Contextualized Embeddings**
   - Each candidate term is embedded using a pretrained BERT model (`bert-base-uncased`).

4. **Clustering**
   - Agglomerative clustering is applied to group semantically similar terms.

5. **Weakly Supervised Filtering**
   - Clusters are filtered using manually constructed seed lists (economics vs. general domain).

6. **Baseline Comparison**
   - A simple TF-IDF–based baseline is implemented for comparison.

---

## Dataset

- **Domain**: Economics
- **Language**: English
- **Source**: OpenStax – *Principles of Economics 3e*
- **License**: CC BY 4.0

Due to size and licensing constraints, the dataset is **not included** in this repository.
Only download links are provided in:
```
demo/data_links.md
```

---

## Repository Structure

```
glossex_final/
│
├── src/
│   ├── glossex/          # Core pipeline (preprocess, saliency, embeddings, clustering, filtering)
│   ├── baselines/        # TF-IDF baseline
│   └── utils/            # Demo utilities
│
├── scripts/              # Reproducible demo / experiment scripts
├── demo/                 # Jupyter demo notebook
├── data/                 # Data directory (ignored by git)
├── requirements.txt
└── README.md
```

---

## Demo

A complete demonstration of the pipeline is provided in:
```
demo/demo.ipynb
```

The notebook:
- loads processed outputs
- visualizes top salient terms
- displays final extracted economics terminology

To reproduce demo outputs programmatically:
```bash
python scripts/make_demo_outputs.py
```

---

## Baseline

A lightweight TF-IDF–style baseline is implemented in:
```
src/baselines/tfidf_baseline.py
```

The baseline ranks terms based on frequency penalized by general-language usage and serves as a comparison to the proposed pipeline.

---

## Installation & Usage

### 1. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run pipeline steps
```bash
python src/glossex/preprocess.py
python src/glossex/embeddings.py
python src/glossex/clustering.py
python src/glossex/filtering.py
```

---

## Notes on Reproducibility

- All experiments are reproducible using the provided scripts.
- Data and generated outputs are excluded from version control by design.
- Randomness is minimized to ensure consistent results.

---

## Course Information

- **Course**: NLP
- **Instructor**: Dr. Ghiassi Rad

---

## Author

**Narges Aliheydari**  
Course Project – Automatic Glossary Extraction

---

## Acknowledgements


This project is inspired by prior work on glossary extraction using contextual embeddings. The implementation and experimental setup are original and adapted specifically for this course assignment.

