# Sentiment_Analysis_on_Product_Reviews

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-yellow)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

## Overview

This project implements a **binary sentiment analysis classifier** for Amazon product reviews, leveraging natural language processing (NLP) and machine learning to categorize customer feedback as positive or negative. Built in Python using libraries like NLTK for text preprocessing and scikit-learn for feature extraction and modeling, it serves as an educational yet production-ready pipeline for analyzing consumer sentiment. The classifier achieves an F1-score of approximately 78-80% on a sampled dataset of ~10,000 reviews, highlighting its effectiveness in handling real-world text data challenges like class imbalance and noisy language. As a result, this repository is ideal for beginners in NLP, data scientists exploring customer analytics, or anyone building recommendation systems powered by feedback analysis.

## Features

- **Text Preprocessing**: Custom pipeline for cleaning, tokenization, stopword removal, and stemming to reduce noise and vocabulary size.
- **Feature Engineering**: TF-IDF vectorization with n-grams for capturing contextual phrases (e.g., "not bad").
- **Model**: Logistic Regression (baseline linear classifier with balanced class weights).
- **Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1) and confusion matrices.
- **Interpretability**: Custom word clouds visualizing top positive/negative words, shaped with a dataset logo mask for visual appeal.

## Dataset

We use the **Consumer Reviews of Amazon Products** dataset, focusing on columns like `reviews.text` (review body) and `reviews.rating` (1-5 stars). 

Download from [Kaggle](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset) and place as `7817_1.csv` in the root.

## Key Concepts Explained

### 1. **Text Preprocessing**
Raw reviews are unstructured and noisy (punctuation, URLs, casing). Preprocessing transforms them into model-ready format:
- **Lowercasing**: Ensures "Love" and "love" are treated identically.
- **Noise Removal**: Regex strips URLs (`re.sub(r'http\S+', '')`) and non-alphabetic chars.
- **Tokenization**: NLTK's `word_tokenize` splits into words (e.g., "I love it!" → ["I", "love", "it", "!"]).
- **Stopword Removal**: Filters common words ("the", "is") using NLTK's English stopwords set (~150 words, ~40-50% of text).
- **Stemming**: PorterStemmer reduces inflections (e.g., "running" → "run") to shrink vocabulary from 100K+ to ~5K terms.
  
**Why?** Reduces dimensionality and focuses on semantic signals. Without it, models overfit to noise. Example: Original: "I LOVE this product!!!" → Processed: "love product".

### 2. **Feature Extraction: TF-IDF Vectorization**
Models require numerical inputs, so we convert text to vectors via **Bag-of-Words (BoW)** enhanced by **TF-IDF**:
- **BoW Basics**: Represents docs as word frequency multisets, ignoring order (simple but effective for short reviews).
- **TF-IDF Formula**: Score = TF (term freq in doc) × IDF (log(total docs / docs with term)). Boosts rare, discriminative words (e.g., "disappointing" in negatives).
- **Implementation**: `TfidfVectorizer(max_features = 3000, ngram_range = (1, 2), min_df = 2)` limits vocab, includes bigrams ("not good"), and prunes rares.
  
**Why?** Raw counts favor long docs; TF-IDF normalizes for importance. Outputs sparse matrices (~99% zeros) for efficiency.

### 3. **Data Splitting**
- **Train-Test Split**: 80/20 via `train_test_split(stratify = y)` preserves class ratios, preventing leakage/overfitting.
- **Concept**: Train learns patterns, test simulates real-world deployment. Stratification combats imbalance.

### 4. **Classification Models**
- **Logistic Regression**: Linear model with sigmoid activation for binary probs. `class_weight = 'balanced'` penalizes majority class errors. Coefficients (`model.coef_`) reveal word influences (e.g., high for "great").
  - **Pros**: Interpretable, fast; assumes linear separability.

**Training/Evaluation**: Fit on train TF-IDF, predict on test. Metrics:
- **Accuracy**: Overall correct (78%).
- **Precision/Recall**: Trade-off for imbalance (89%/72%).
- **F1-Score**: Harmonic mean (80%)-key for uneven classes.
- **Confusion Matrix**: Visualizes TP/TN vs. FP/FN.

### 5. **Interpretability: Word Clouds**
- Extracts top 100 words via sorted coefficients.
- `WordCloud` with mask (logo PNG -> binary array) shapes visuals; colormaps (autumn/gray) differentiate sentiments.
- **Why?** Reveals biases (e.g., "love" positive, "waste" negative)-debugs/tunes models.

## Results

- **Performance** (Test Set, n = 320):
  | Model              | Accuracy | Precision | Recall | F1-Score |
  |--------------------|----------|-----------|--------|----------|
  | Logistic Regression| 0.78    | 0.89     | 0.72  | 0.80    |

- **Confusion Matrix Example**: High true positives (positives correctly ID'd), but some false negatives (missed negatives-tune recall via thresholds).
- **Top Words**: Positives: "love", "recommend", "soft"; Negatives: "disappoint", "cheap", "broke".
- **Limitations**: Bag-of-Words ignores order/sarcasm; future: BERT for context.

## License

MIT License Copyright (c) 2025 *1453nicat*
