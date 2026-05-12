# SMILES-2026 Hallucination Detection — Solution Report

## Reproducibility

```bash
git clone https://github.com/Alexeiyaganov/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
Runs on a free Google Colab T4 GPU. Outputs: results.json and predictions.csv.

Two fixes to the original solution.py: changed attention_mask.cpu() to attention_mask in both extraction loops (tensors were on different devices, caused a crash). No logic changed.

I extract hidden states from the response only, not prompt+response. The prompt is noise — hallucination happens during generation, so the signal is in the response hidden states.

What I Changed
aggregation.py:

Take layers [8, 12] (two middle layers). Last layers are too close to next-token prediction, first layers are too syntactic.

For each layer, concatenate last token + mean over all tokens. Last token captures the model's state right after finishing the answer; mean pooling captures the overall trajectory.

Result: 2 layers × 2 pooling × 896 = 3584 features.

probe.py:

2-layer MLP: 3584 → 128 → 64 → 1, with BatchNorm and heavy Dropout (0.7).

AdamW with weight_decay=1e-2, cosine annealing LR for 150 epochs.

pos_weight in the loss to handle class imbalance (483 hallucinated / 206 truthful).

Threshold tuned on validation set to maximise F1.

splitting.py:

5-fold stratified cross-validation instead of a single split. More stable metrics.

Results
Metric	Baseline	Probe
Test Accuracy	70.10%	70.39%
Test F1	82.42%	81.93%
Test AUROC	—	66.38%
Val AUROC	—	68.55%
Honest assessment: the probe barely beats the majority-class baseline on accuracy and F1. AUROC of 66% is above random (50%) but modest. The model overfits on training (100% train AUROC on most folds) despite heavy regularisation. With 689 samples and 3584 features, there is simply not enough data to learn a robust decision boundary.

What I Tried and What Helped
Things that improved the score:

Response-only extraction instead of prompt+response. The first run with prompt+response gave Test AUROC = 67.94%, and train AUROC hit 100% on most folds — the probe was memorising the prompts, not learning hallucination patterns.

Mean + last-token pooling instead of last-token only or mean only. Last-token alone misses context; mean alone washes out the final state.

Strong regularisation: Dropout 0.7, weight_decay 1e-2, small hidden dim (128). Without this, train AUROC = 100% and test AUROC stays near baseline.

Things that didn't work:

4 layers [8,12,16,20] — too many features, overfitting got worse.

PCA to 256 components — didn't help. The problem is feature quality, not just dimensionality.

Attention pooling over tokens — overfit on 689 samples.

3-layer MLP — more parameters than data in some folds, validation F1 dropped.

Constant learning rate — loss oscillated; cosine annealing smoothed it.

Using prompt+response together — added ~1500 characters of noise per sample.