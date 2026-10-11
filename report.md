# Report

Q1. The dominant confusion pairs are truck→automobile (1,247) and automobile→truck (1,126), followed closely by cat→dog (924) and dog→cat (834). These reflect systematic annotator mistakes between visually similar object categories. Deer→horse (683) and deer→dog (491) are next. Together these five pairs account for roughly 30% of all label errors.

Q2. Memorization starts at epoch 38. Evidence: the 3-point smoothed train_acc_mislabeled stays flat below 5% through epoch 37, then rises sharply and stays above baseline + 5% for 3+ consecutive epochs. val_acc_clean peaks at epoch 36 (within 2 epochs of this), confirming the network transitions from learning true patterns to fitting noise.

Q3. Five-fold out-of-fold CE loss (−log p(noisy_label)) worked best (AUROC 0.8910, combined 0.8919). Logit-space margin added complementary information (AUROC 0.8922). No per-class normalization was used — horse has 28% noise, cat has 51.5%; equalizing them would hurt precision@k. Failure mode: ambiguous-but-correctly-labeled images (e.g. a blurry cat) score high because the model is genuinely uncertain — ambiguity ≠ mislabeling.

Q4. Co-teaching achieved the best validation accuracy on noisy labels at 57.72%, outperforming SCE (57.24%) and filter+retrain (56.50%). SCE's reverse-CE term penalizes gradient updates toward labels the model is uncertain about, reducing memorization. Co-teaching uses network disagreement to avoid co-memorizing the same errors. Filter+retrain removes 40% of training data, discarding hard-but-correct examples along with noise.

Q5. Under 40% synthetic uniform noise, co-teaching achieved 53.52% vs 43.74% for standard CE (+9.78% gain). On human noise, co-teaching reached 57.72% vs 50.94% (+6.78% gain). Human noise is class-structured (concentrated in 2-3 visually similar pairs), so baseline CE retains higher accuracy since errors stay semantically close. Synthetic uniform noise scatters across all classes, causing worse degradation for CE, while co-teaching effectively filters random outliers regardless of noise structure.
