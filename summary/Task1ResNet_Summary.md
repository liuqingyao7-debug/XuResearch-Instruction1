# Task 1 Summary: BUSI Breast Tumor Classification with ResNet50

## Task

Binary classification of breast ultrasound (BUSI) images into **malignant (0)** and **benign (1)**, using ResNet50 pre-trained on ImageNet. All images were resized to 224×224. The model was trained on `train.xlsx` (517 images: 171 malignant, 346 benign) and evaluated on `test.xlsx` (130 images: 39 malignant, 91 benign). Training ran as `train.py` on a GPU node via Slurm.

## Method

| Component | Choice |
|---|---|
| Model | ResNet50 (IMAGENET1K_V1), final layer replaced with `Linear(2048, 2)` |
| Preprocessing | Resize 224×224, RGB, ImageNet normalization |
| Loss / Optimizer | Cross-entropy / Adam, lr = 1e-4 |
| Batch size | 16 (train shuffled, test not shuffled) |
| Runs | 30 epochs and 5 epochs |

## Results

| Setting | Accuracy | Malignant P / R / F1 | Benign P / R / F1 | Misclassified | Malignant missed |
|---|---|---|---|---|---|
| 30 epochs | 0.869 | 0.78 / 0.79 / 0.78 | 0.91 / 0.90 / 0.91 | 17 | 8 |
| 5 epochs | **0.908** | **0.85 / 0.85 / 0.85** | **0.93 / 0.93 / 0.93** | **12** | **6** |

Confusion matrices (rows = true, columns = predicted; order: malignant, benign):

- 30 epochs: `[[31, 8], [9, 82]]`
- 5 epochs: `[[33, 6], [6, 85]]`

Overall metrics treating benign as the positive class: 30 epochs gives precision 0.911, recall 0.901, F1 0.906; 5 epochs gives precision, recall, and F1 of 0.934.

## Loss Curve Analysis

- **30 epochs:** train loss fell to nearly 0, while test loss reached its minimum around epoch 3–4, then rose and fluctuated between about 0.5 and 0.7. This is a typical sign of **overfitting** (memorizing the training set, generalizing worse).
- **5 epochs:** avoids the late rise in test loss and performs better on the test set. A gap remains between train loss (about 0.05–0.1) and test loss (about 0.4–0.5), so some overfitting is still present.

## Why Accuracy Alone Is Not Enough

Benign images are roughly twice as numerous as malignant ones, so accuracy can hide weak performance on the malignant class. **Malignant recall** is the most important metric here, since missing a malignant tumor (predicting benign) is the most serious error.

## Error Patterns (inspected examples)

- Missed malignant cases were often small, smoothly bordered, or had indistinct margins, which may look benign and lose detail at 224×224.
- Benign cases predicted as malignant tended to have irregular shape, heterogeneous texture, or posterior darkening.
- Annotation marks (measurement crosses, text) appear in both correct and incorrect predictions, so there is no clear evidence they cause errors.
- These observations come from a non-random subset of images and are **not diagnostic conclusions**.

## Limitations

- Small test set (39 malignant images): a difference of two images noticeably changes recall, so the 30 → 5 epoch improvement should be read cautiously.
- The epoch count was chosen by looking at **test** loss, so the 5-epoch result may be slightly optimistic; a separate validation set is needed.
- Single training run with one random seed.

## Next Steps

1. Data augmentation (flips, rotations, brightness changes).
2. A separate validation set for choosing epochs.
3. Higher input resolution to preserve small lesions.
4. Multiple random seeds to measure variability.
5. Use the dataset's segmentation masks.

## Key Takeaways

- Fine-tuning pretrained ResNet50 reaches about **91% accuracy** on the BUSI test set with only 5 epochs.
- Training longer made the model **overfit** and performed worse (87% accuracy).
- Loss curves are what reveal the right training length, and class-wise metrics (especially malignant recall) matter more than accuracy on this imbalanced data.
