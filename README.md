# Movie Review Sentiment Analysis

## Overview
This project builds a supervised binary text classification model that labels IMDB movie reviews as positive or negative from their text alone. Raw reviews are cleaned and normalized, converted to numerical features with TF-IDF, and used to train and compare four classical machine learning models. The goal is to identify the model that most reliably separates positive from negative sentiment on unseen data.

## Dataset
The project uses the IMDB Dataset of 50K Movie Reviews, loaded via `kagglehub` from Kaggle (`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`).

| Property | Detail |
|---|---|
| Total reviews | 50,000 |
| Columns | `review` (text), `sentiment` (label) |
| Classes | positive, negative |
| Class balance | 25,000 positive / 25,000 negative |
| Missing values | None |
| Duplicates | 418 (removed before modeling) |

The dataset is perfectly balanced, which keeps accuracy a trustworthy metric and removes the need for resampling.

## Methodology
1. Exploratory data analysis: class distribution, review length, word and character counts, most frequent words, and word clouds per sentiment.
2. Feature engineering: review length, word count, uppercase ratio, exclamation and question counts, and average word length.
3. Text cleaning: contraction expansion, lowercasing, removal of HTML tags, URLs, and numbers, collapsing of repeated words and characters, and removal of special characters.
4. Tokenization and normalization: word tokenization, stopword removal with negation words preserved, and POS-aware lemmatization.
5. Vectorization: TF-IDF with `max_features=5000`.
6. Modeling: an 80/20 train-test split (`random_state=42`), training four classifiers, and validating the top model with 5-fold cross-validation.

## Results
Model performance on the held-out test set:

| Model | Accuracy | Recall | Precision | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.8819 | 0.8945 | 0.8733 | 0.8838 |
| Ridge Classifier | 0.8783 | 0.8915 | 0.8693 | 0.8803 |
| Naive Bayes | 0.8470 | 0.8534 | 0.8437 | 0.8485 |
| Random Forest | 0.8356 | 0.8305 | 0.8402 | 0.8353 |

Logistic Regression is the best-performing model across every metric, and its 5-fold cross-validation score confirms the result is stable rather than a product of the split. Because the classes are balanced, F1 is used as the headline metric, with accuracy reported alongside it.

## Tech Stack
| Category | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn, wordcloud |
| Text processing | nltk, BeautifulSoup (bs4), contractions, re |
| Machine learning | scikit-learn (TF-IDF, Logistic Regression, Ridge Classifier, Multinomial Naive Bayes, Random Forest) |
| Data source | kagglehub |
