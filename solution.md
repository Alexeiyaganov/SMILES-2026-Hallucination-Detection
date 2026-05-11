# SMILES-2026 Hallucination Detection — Solution Report

## 1. Reproducibility

### Environment
- Python 3.10+
- Dependencies from `requirements.txt`
- GPU: NVIDIA T4 (Google Colab free tier) or any CUDA GPU with ≥4 GB VRAM

### Setup & Run
```bash
git clone https://github.com/Alexeiyaganov/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
Output
results.json — accuracy, F1, AUROC averaged across folds

predictions.csv — predicted labels for data/test.csv

Notes on reproducibility
Hidden states are extracted by feeding prompt + response into Qwen2.5-0.5B with output_hidden_states=True.

I had to change two lines in solution.py: attention_mask.cpu() → attention_mask in both extraction loops. Without this, the mask ends up on CPU while hidden states are on GPU, and it crashes. No logic changes — same tensors, same device.

Extraction uses batch size 4 to stay within T4 memory.

2. Final Solution Description
Aggregation (aggregation.py)
Which layers I used: [8, 12, 16, 20] — four middle layers, concatenated. I tried using just the last layer (the default), and the probe barely beat the baseline. Middle layers carry more semantic information; the last layer is too focused on next-token prediction. I also tried all 24 layers, but that gave 21504 features and the probe overfit horribly.

Token pooling: Mean over all real tokens. The default was taking only the last token, which throws away most of the response. Hallucination cues can show up anywhere in the generated text, so mean pooling made intuitive sense. I tested attention-weighted pooling too, but on only 689 samples it overfit — mean pooling was simpler and generalised better.

Feature dimension: 4 layers × 896 = 3584 features.

Geometric features (turned off by default, flip USE_GEOMETRIC=True to enable):

Per-layer L2 norm of the mean vector (24 features — one per layer)

Per-layer standard deviation averaged across dimensions (24 features)

Cosine similarity between the mean representation of layer 8 and layer 20 (1 feature)

The idea was to capture how "spread out" the representations are and how much they drift across layers. In my runs, enabling these gave a small boost to validation AUROC but increased feature extraction time, so I left them off for the final submission. They're in the code if the reviewer wants to test.

Probe (probe.py)
Architecture: input → 512 → 256 → 1, with BatchNorm and Dropout(0.3) after each hidden layer.

Why this and not the default single-layer 256 → 1:

The default MLP was underfitting. Two layers with BatchNorm learn a better decision boundary.

Dropout helps because the dataset is small — 689 samples with 3584 features is a recipe for overfitting otherwise.

Pos weight in BCEWithLogitsLoss handles the imbalance (483 hallucinated vs 206 truthful). Without it, the model was biased toward predicting "hallucinated" for everything.

Training: AdamW with weight decay 1e-4, cosine annealing from 1e-3 to 1e-5 over 300 epochs. I started with plain Adam and constant LR, but the loss was jumping around. Cosine annealing smoothed it out.

Threshold tuning: after training, I sweep over possible thresholds on the validation set and pick the one that maximises F1. The default 0.5 is not always best, especially with imbalanced classes.

Splitting (splitting.py)
5-fold stratified cross-validation. Each fold: 20% test, then the remaining 80% is split 85/15 into train/val. Stratification keeps the class ratio consistent across splits.

I started with a single 70/15/15 split (the default). Metrics were jumping ±5% between runs depending on the random seed. 5-fold averages this out and gives a more honest estimate.

What actually helped
The initial run with 4 layers [8,12,16,20] and 3584 features gave **Test AUROC = 67.94%** — barely above the majority-class baseline. The probe was memorising the training set (100% train AUROC on most folds) and failing to generalise.

After seeing these results, I made several changes:

1. **Reduced layers from 4 to 2** ([16, 20] instead of [8,12,16,20]). Feature dimension dropped from 3584 to 1792.
2. **Increased Dropout from 0.3 to 0.5** and reduced hidden dim from 512 to 256 — fewer parameters, stronger regularisation.
3. **Increased weight decay from 1e-4 to 1e-3.**
4. **Added PCA reduction** to 256 components before the MLP. This was the single biggest improvement — it forced the model to learn a compressed representation instead of memorising noise.
5. **Reduced epochs from 300 to 150** to prevent overfitting.

These changes closed the gap between train and validation AUROC significantly.

3. Experiments and Failed Attempts
Ideas I tried and discarded:
All 24 layers. 21504 features → probe overfit immediately. Train accuracy 99%, test accuracy barely above baseline. Waste of time, but at least now I know.

Attention-weighted pooling. I wrote a simple attention mechanism over tokens. On paper it should be better than mean pooling because the model learns which tokens matter. In practice, on 689 samples, it just memorised noise. Mean pooling was simpler and worked.

3-layer MLP. 3584 → 512 → 256 → 128 → 1. More parameters than training examples in the smaller folds. Validation F1 was worse than 2-layer. Dropped it.

Single split instead of 5-fold. The default in the skeleton code. I ran it three times with different seeds and got test accuracy 68%, 74%, 71%. Too noisy to trust. 5-fold gives consistent numbers.

No pos_weight in the loss. The model learned to predict "hallucinated" almost every time, because that's the majority class. Accuracy looked okay (~70%) but F1 was terrible. Adding pos_weight based on the actual class ratio fixed this.

Constant learning rate. With lr=1e-3 constant, the loss curve was noisy and the final validation AUROC was lower than with cosine annealing. Not a huge difference, but consistent.

**First run — 4 layers, no PCA.** 3584 features on 447 training samples is too many. The probe achieved 100% train AUROC and ~68% test AUROC, which is barely above baseline. Clear overfitting. This motivated the dimensionality reduction approach.

4. Link to Predictions
predictions.csv is in the repository root:
https://github.com/Alexeiyaganov/SMILES-2026-Hallucination-Detection/blob/main/predictions.csv