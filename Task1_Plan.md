# Task 1 Plan: BUSI Breast Tumor Classification (ResNet50)

Based on *Research Training Instructions* and the completed `Task1.ipynb`.

## 1. Goal

Train a deep learning model to classify breast ultrasound (BUS) images as **benign** or **malignant**, using **ResNet50** with ImageNet pre-trained weights. Deliver a well-organized notebook (a text cell before every code cell, comments in the code) and understand every line of it.

## 2. Setup and Workflow Tips

| Item | Plan |
|---|---|
| Planning | Use a chatbot (ChatGPT / Claude AI) to summarize literature and plan directions; use Codex / Claude Code only for implementation planning and coding |
| Folders | `~/datasets/` for data, `~/projects/research_training/` for code and outputs |
| Long jobs | Never use the notebook Run button for deep learning (VPN disconnects every 30–60 min) |
| GPU server | `nohup python train.py > train.log 2>&1 &` |
| HPC (e.g., DGX Spark) | Submit with Slurm: `sbatch submit.sh` |
| Data transfer | Download BUSI locally, upload via command line or FileZilla |

## 3. Background Concepts to Study First

- `.py` file vs `.ipynb` notebook
- Deep learning, computer vision, image classification vs segmentation
- Data and labels; train / validation / test split and the purpose of each
- CNN components: convolution, pooling, batch normalization, activation function, fully connected layer
- Training concepts: forward / backward propagation, epoch, loss function, optimizer, learning rate, batch size, overfitting vs underfitting

## 4. Step-by-Step Implementation Plan

1. **Warm-up:** create `research_training/Test.ipynb`, run a simple Python program, practice text + code cells.
2. **Environment check:** confirm PyTorch, torchvision, pandas, openpyxl, scikit-learn, matplotlib, and GPU visibility.
3. **Explore the dataset:** list the `images/`, `labels/`, `train.xlsx`, `test.xlsx` contents.
4. **Inspect spreadsheets:** check columns (`Image`, `Label`) and shapes.
5. **Check label distribution:** verify what 0 and 1 mean (0 = malignant, 1 = benign) by cross-checking against filenames.
6. **Verify images exist:** every filename in the spreadsheets must be present in `images/`.
7. **Dataset and DataLoader:** custom `BUSIDataset`, resize to **224×224**, convert to RGB, normalize with ImageNet mean/std; shuffle train only.
8. **Build the model:** `resnet50` with `IMAGENET1K_V1` weights; replace `fc` with `Linear(2048, 2)`.
9. **Loss, optimizer, functions:** `CrossEntropyLoss`, `Adam (lr=1e-4)`, `train_one_epoch()` and `evaluate()`.
10. **Train on GPU via Slurm:** `train.py` + `submit.sh`; save `history.json`, `resnet50_busi.pth`, `test_predictions.csv`. Run 30 epochs, then 5 epochs.
11. **Plot loss curves:** training loss vs testing loss per epoch; judge under/overfitting and a good epoch count.
12. **Evaluate:** accuracy, precision, recall, F1, confusion matrix, per-class report.
13. **Visualize predictions:** loop over every test image showing the true and predicted class (wrong ones in red).
14. **Write summary and analysis:** results, loss curves, error patterns, limitations, next steps.

## 5. Notebook Structure

| # | Section |
|---|---|
| 0 | Title and Python check |
| 1 | Environment check |
| 2–5 | Dataset exploration and sanity checks |
| 6 | Dataset and DataLoader |
| 7 | Model |
| 8 | Loss, optimizer, train/eval functions |
| 9 | GPU training (`train.py` + Slurm) |
| 10 (a/b) | Loss curves, 30 and 5 epochs |
| 11 (a/b) | Evaluation metrics, 30 and 5 epochs |
| 12 | Test prediction visualization |
| 13 | Summary and analysis |

## 6. Key Configuration

| Parameter | Value |
|---|---|
| Model | ResNet50, ImageNet pretrained |
| Input size | 224 × 224 × 3 |
| Normalization | mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] |
| Loss | Cross-entropy |
| Optimizer | Adam, lr = 1e-4 |
| Batch size | 16 |
| Epochs | 30, then 5 |
| Train / Test | 517 / 130 images |

## 7. Deliverables Checklist

- [x] `Test.ipynb` warm-up notebook
- [x] `Task1.ipynb` with text cell before each code cell and comments
- [x] Pretrained ResNet50 fine-tuned on `train.xlsx`, evaluated on `test.xlsx`
- [x] Loss curve plots
- [x] Accuracy, precision, recall, F1, confusion matrix
- [x] Loop showing each test image with true and predicted class
- [x] Summary and analysis
- [ ] Be able to explain every line of code and the background concepts

## 8. Next Steps (Future Improvements)

1. Add data augmentation (flips, rotations, brightness).
2. Hold out a separate **validation set** to choose the number of epochs.
3. Try higher input resolution to keep small lesions visible.
4. Repeat with several random seeds to measure variability.
5. Explore the segmentation masks in the dataset (links to the segmentation task).
