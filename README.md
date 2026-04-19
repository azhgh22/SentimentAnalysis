# Sentiment Analysis on Movie Reviews

This project trains and compares three deep learning architectures for 5-class sentiment classification on the Kaggle competition dataset `sentiment-analysis-on-movie-reviews`.

The training and evaluation workflow is implemented in `models.ipynb`.

## Models Compared

The notebook trains these three models on the same train/validation split and reports validation performance:

1. `SentimentRNN` (Bidirectional RNN)
2. `SentimentLSTM` (Bidirectional LSTM)
3. `TransformerSentiment` (Custom Transformer encoder)

## Dataset

- Source: Kaggle competition `sentiment-analysis-on-movie-reviews`
- Input: movie review phrases
- Target: `Sentiment` with 5 classes (`0..4`)
- Split used in notebook: `train_test_split(..., test_size=0.2, random_state=42)`

## Shared Training Setup

From notebook config (`models.ipynb`):

- `VOCAB_SIZE = 20000`
- `MAX_LEN = 50`
- `EMBEDDING_DIM = 100`
- `HIDDEN_DIM = 128`
- `NUM_LAYERS = 3` (RNN/LSTM)
- `NUM_CLASSES = 5`
- `BATCH_SIZE = 64`
- `EPOCHS = 50`
- `LR = 1e-4`
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (`patience=2`, `factor=0.5`)

Tokenization uses Keras `Tokenizer` and `pad_sequences` with post-padding/truncation.

## Architecture Notes

### 1) Bidirectional RNN (`SentimentRNN`)

- Embedding initialized from GloVe `glove-wiki-gigaword-100`
- Embedding is frozen (`requires_grad = False`)
- Multi-layer bidirectional `nn.RNN` (`tanh`)
- Uses packed sequences (`pack_padded_sequence`) to handle variable lengths
- Final representation: concat of last forward/backward hidden states

### 2) Bidirectional LSTM (`SentimentLSTM`)

- Embedding initialized from same GloVe vectors
- Embedding is trainable (`requires_grad = True`)
- Multi-layer bidirectional `nn.LSTM`
- Also uses packed sequences
- Final representation: concat of last forward/backward hidden states

### 3) Transformer (`TransformerSentiment`)

- Learnable token embeddings + sinusoidal positional encoding
- Custom multi-head self-attention
- Transformer blocks with residual connections, layer norm, and feed-forward layers
- Sequence pooling by mean over time dimension

## Train/Evaluation Results (from notebook outputs)

The notebook prints `Best val accuracy` for each model.

| Model | Best Validation Accuracy |
|---|---:|
| BiRNN | `0.6689` |
| BiLSTM | `0.6795` |
| Transformer | `0.6512` |

## Comparison and Interpretation

1. **Best overall: BiLSTM (`0.6795`)**
	- Highest validation accuracy among the three models.
	- Likely benefits from LSTM gating and bidirectional context for phrase-level sentiment.

2. **Second: BiRNN (`0.6689`)**
	- Close to LSTM but lower by about `1.06` accuracy points.
	- Simpler recurrent dynamics can underperform LSTM on longer or more nuanced phrases.

3. **Third: Transformer (`0.6512`)**
	- Lowest validation accuracy in this exact setup.
	- Suggests the current Transformer configuration/training regime needs tuning to match recurrent baselines on this dataset.

### Practical Note: Why RNN Can Still Be a Best Choice

In this experiment, the Transformer appears to have higher training variance than the recurrent models, even when dropout is kept the same (`DROPOUT = 3.0` in the shared config). In other words, regularization level alone does not explain the gap.

## Final Ranking

1. `BiLSTM` (`0.6795`)
2. `BiRNN` (`0.6689`)
3. `Transformer` (`0.6512`)

## Reproduce

1. Open `models.ipynb`.
2. Run cells in order.
3. Train all three models.
4. Compare the printed `Best val accuracy` values.

## Output

The notebook also generates `submission.csv` using model predictions on the competition test set.

I uploaded the RNN submission to Kaggle because it was the most stable model with lower variance, and it got `0.64` test accuracy.