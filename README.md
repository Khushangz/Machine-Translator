
# Machine Translation Project

This project implements a machine translation model to translate between English and German using deep learning techniques. The notebook walks through data preprocessing, model training, and evaluation using TensorFlow and other necessary libraries.

## Project Overview

- **Language Pair**: English-German
- **Approach**: Sequence-to-sequence model with neural networks
- **Objective**: Translate sentences from German to English using a trained deep learning model

## Project Structure

1. **Data Loading**:
   - The dataset (`deu.txt`) is loaded and split into English-German sentence pairs.
   - The `load_doc()` function is used to load the dataset, which is cleaned using various text preprocessing methods (removing punctuation, lowercasing, etc.).

   ```python
   filename = 'deu.txt'
   doc = load_doc(filename)
   pairs = to_pairs(doc)
   clean_pairs = clean_pairs(pairs)
   ```

2. **Preprocessing**:
   - The text is normalized by removing non-printable characters, punctuation, and converting all characters to lowercase.
   - Sentence pairs are tokenized and cleaned before feeding into the model.

3. **Model Architecture**:
   - A sequence-to-sequence (Seq2Seq) model is implemented using a deep learning architecture.
   - Both encoder and decoder networks are used to learn translation patterns between English and German sentences.

4. **Training and Evaluation**:
   - The model is trained on a preprocessed dataset, and performance is evaluated using BLEU scores.
   - BLEU scores for 1-gram, 2-gram, 3-gram, and 4-gram translations are printed to evaluate the accuracy of the translations.

   ```python
   print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
   print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
   print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
   print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
   ```

## Requirements

- **Python 3.x**
- **TensorFlow**
- **Numpy**
- **Pickle**
- **NLTK** (for BLEU score evaluation)

## Usage

1. Install the required dependencies:

   ```bash
   pip install tensorflow numpy nltk
   ```

2. Prepare the dataset (`deu.txt`) by placing it in the working directory.

3. Run the notebook to preprocess the data, train the model, and evaluate its performance using BLEU scores.

4. Model predictions can be used to translate German sentences into English.

## Dataset

- The dataset used is a simple tab-separated file (`deu.txt`) containing pairs of English and German sentences.

## Evaluation

- The model's performance is evaluated using the BLEU (Bilingual Evaluation Understudy) score, which measures the quality of the translated sentences compared to reference translations.

