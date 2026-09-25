# Training

> **PictureStudio v2.5.2** — Train a CNN model locally or remotely via SSH

---

# Training

Train a CNN model – locally or on a GPU server via SSH.

## Configuration

| Parameter | Recommendation | Description |
|---|---|---|
| Architecture | ResNet-18 to start | Model architecture |
| Epochs | 20–30 | Number of training passes |
| Learning rate | 0.001 | Step size for weight updates |
| Batch size | 32 (GPU) / 8–16 (CPU) | Images per training step |
| Device | auto | Automatic: GPU > MPS (Apple) > CPU |
| Early Stopping | 5–7 | Stop after N epochs without improvement |
| LR Scheduler | cosine | Automatically adjust the learning rate |
| Class balancing | when imbalanced | Equalize via WeightedRandomSampler |

## Hyperparameter Search (optional)

Click **⚙ Hyperparameter-Suche…** (`pip install optuna` required).
Optuna automatically tests combinations of learning rate, batch size, architecture, and optimizer.

### Live Progress Display

While the search is running, a dialog shows:

- **Progress bar** — Trial X / Total
- **Status line** — current best Val-Accuracy
- **Scrolling log** — one line per completed trial with parameters and result

```
[Trial  1/20]  lr=3.2414e-03  batch=16  model=resnet18  opt=adam  →  Acc: 84.21%  ★ New best!
[Trial  2/20]  lr=8.1450e-04  batch=32  model=resnet50  opt=sgd   →  Acc: 79.05%
```

A **★** marks each new best result. The button switches from *Abbrechen* (Cancel) to *Schließen* (Close) when finished.

Best parameters are applied directly to the training configuration.

## Starting & Monitoring Training

> 💡 **Button order:** ① Hyperparameter-Suche → ② Training starten → ③ Training stoppen

**Start training**
Click *Training starten*. Live display shows:

- **Train-Loss** and **Val-Loss** – ideally decrease together
- **Train-Acc** and **Val-Acc** – ideally increase together

The best model (highest Val-Acc) is saved automatically.

> ⚠️ **Recognizing overfitting:** Train-Loss decreases while Val-Loss increases → the model is memorizing the training data. Fix: add more data, enable Early Stopping, use stronger augmentation.

## SSH Remote Training

**Setup**
1. *Settings → SSH profiles* – create a profile (host, user, key path)
2. Training page: enable the SSH checkbox → select a profile → test the connection
3. Start training → the app zips the data, uploads it, streams logs, and automatically downloads the best model.

## After Training

**Generate reports**

- *HTML report* – full report with loss/accuracy curves and confusion matrix
- *Excel report* – metrics as a table for documentation
