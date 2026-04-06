# Deep Learning Homeworks — Image Classification & Speech Recognition

Two deep learning assignments implementing and benchmarking neural network architectures from scratch and using PyTorch, applied to image classification (OCTMNIST) and sequence-to-sequence speech recognition.

---

## Contributors

| Name | GitHub |
|---|---|
| Tomás Marques | [@TomasMarques175](https://github.com/TomasMarques175) |
| Apolónia | [@apolonia-p](https://github.com/apolonia-p) |

---

## Homework 1 — Linear Models & MLP from Scratch (NumPy + PyTorch)

### Overview

Implements and trains three classification models on the **OCTMNIST** dataset (retinal OCT image classification, 4 classes) — first from scratch in NumPy, then re-implemented in PyTorch with mini-batch SGD.

### Dataset

**OCTMNIST** — 28×28 greyscale retinal OCT images across 4 diagnostic classes, loaded via `octmnist.npz`.

### Models Implemented

**Question 1 — NumPy from scratch:**

- **Perceptron** — multiclass perceptron with online weight updates
- **Logistic Regression** — softmax + cross-entropy loss, SGD update rule
- **MLP (2-layer)** — manual forward pass, backpropagation, and ReLU activation implemented in pure NumPy

**Question 2 — PyTorch:**

- **Logistic Regression** — `nn.Linear` with `CrossEntropyLoss`
- **Feedforward Network (MLP)** — configurable depth, hidden size, activation (ReLU/Tanh), dropout, L2 regularisation, SGD/Adam optimisers

### Experiments & Hyperparameter Tuning

Training and validation curves were generated across multiple configurations:
- Learning rates: `0.001`, `0.01`, `0.1`, `1.0`
- Batch sizes: `16`, `256`, `1024`
- Dropout: `0.0`, `0.2`, `0.3`
- L2 decay, epochs up to `150`

### Usage

```bash
# NumPy models
python hw1-q1.py perceptron
python hw1-q1.py logistic_regression
python hw1-q1.py mlp -learning_rate 0.001 -hidden_size 200 -epochs 20

# PyTorch models
python hw1-q2.py logistic_regression -batch_size 16 -learning_rate 0.01 -epochs 20
python hw1-q2.py mlp -batch_size 256 -learning_rate 0.1 -layers 2 -hidden_size 200 -dropout 0.2 -optimizer adam
```

---

## Homework 2 — CNNs & Sequence-to-Sequence with Attention (PyTorch)

### Overview

Extends to two more advanced deep learning tasks: image classification with convolutional networks and speech-to-text with encoder-decoder architectures using RNN and Transformer decoders with attention.

### Question 2 — Convolutional Neural Network (CNN)

Implements a CNN for OCTMNIST classification with two configurations:

- **Q2.1 — CNN with Max-Pooling:** Conv(1→8, 3×3) → MaxPool → Conv(8→16, 3×3) → MaxPool → FC(576→320) → Dropout → FC(320→120) → FC(120→4)
- **Q2.2 — CNN with Strided Convolutions (no max-pool):** same architecture but uses stride=2 convolutions instead of pooling layers

Includes `get_number_trainable_params()` to count and compare model parameters between both variants.

```bash
python hw2-q2.py -learning_rate 0.01 -dropout 0.7 -optimizer sgd
python hw2-q2.py -learning_rate 0.01 -dropout 0.7 -optimizer sgd -no_maxpool
```

### Question 3 — Speech-to-Text with Attention (Notebook)

Implements a full **sequence-to-sequence speech recognition** pipeline trained on a speech dataset (10-second audio clips → text transcripts):

- **Encoder:** processes mel-spectrogram audio features (`80 mel filters × 2754 time steps`)
- **Decoder (a) — RNN with Attention:** GRU-based decoder with attention mechanism over encoder outputs
- **Decoder (b) — Transformer:** multi-head self-attention Transformer decoder

The model is evaluated using **text distance metrics** (character-level edit distance) via the `textdistance` library.

Designed to run on **Google Colab or Kaggle** with GPU acceleration (T4).

---

## Project Structure

```
Homework1/
├── skeleton-code/
│   ├── hw1-q1.py          # NumPy: Perceptron, Logistic Regression, MLP
│   ├── hw1-q2.py          # PyTorch: Logistic Regression, MLP
│   ├── utils.py           # Data loading and seeding utilities
│   ├── octmnist.npz       # OCTMNIST dataset
│   └── Figures/           # Generated training loss and validation accuracy plots

Homework2/
├── skeleton_code/
│   ├── hw2-q2.py          # PyTorch CNN (with and without max-pooling)
│   ├── hw2-q3.ipynb       # Seq2Seq Speech-to-Text with RNN/Transformer decoder
│   ├── utils.py           # Data loading utilities
│   └── storage/           # Saved model checkpoints and metrics
```

---

## Requirements

```bash
pip install numpy torch torchvision matplotlib ml_collections "textdistance[extras]"
```

For the speech notebook, GPU access via Colab or Kaggle is recommended.

---

## Key Concepts Covered

| Concept | Where |
|---|---|
| Perceptron & online learning | HW1 Q1 |
| Softmax + cross-entropy, SGD | HW1 Q1 & Q2 |
| Manual backpropagation (NumPy) | HW1 Q1 MLP |
| Mini-batch training, dropout, L2 reg | HW1 Q2 |
| Convolutional layers, max-pooling vs. strided conv | HW2 Q2 |
| Encoder-decoder architecture | HW2 Q3 |
| RNN decoder with attention | HW2 Q3(a) |
| Transformer decoder (multi-head attention) | HW2 Q3(b) |
| Mel-spectrogram audio features | HW2 Q3 |
