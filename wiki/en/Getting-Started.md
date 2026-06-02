# Getting Started

> **PictureStudio v2.3.0** — Complete workflow from your first images to a trained model

---

# Getting Started – Complete Workflow

This guide walks you through the entire process, from your first image to a trained model.

---

## Phase 1 – Create a Project

**Step 1 – New Project**
Menu *File → New Project* (`Ctrl+N`). Enter a meaningful name and choose the project type:

- 📸 **Image Classification** – classifies individual images into classes
- 🎬 **Video Analysis & Anomaly Detection** – live camera stream with autoencoder-based detection

The project is saved as a `.json` file.

---

## Phase 2 – Load Images

**Step 2 – Add an Image Folder**
Go to the **Data page** → click *Load images…* and select a folder.
All `.jpg`, `.png`, `.bmp`, and `.tiff` files are added automatically. Drag & drop into the window also works.

**Step 3 – Analyze the Dataset**
Click *Analyze dataset* to check for:

- **Missing files** – images that can no longer be found
- **Duplicates** – identical images (MD5 hash)
- **Class imbalance** – poor training results when one class has far more images than others

---

## Phase 3 – Define Labels & Annotate Images

**Step 4 – Create Labels**
Go to the **Labeling page** → *Project → Manage labels…* (`Ctrl+L`).
Add a label for each class (e.g. "good", "defective", "unclear").

**Step 5 – Label Images**
1. Click an image in the thumbnail list
2. Press `1`–`9` for quick label assignment
3. Press `N` for the next image, `P` for the previous

**Goal:** At least 50–100 images per class. For good models: 200+ per class.

**Step 6 – Draw ROIs (optional)**
If you only want to analyze a specific area of the image:

- `R` = Rectangle  &nbsp; `E` = Ellipse  &nbsp; `G` = Polygon

Select an ROI in the right-hand list → assign a label.

---

## Phase 4 – Configure & Start Training

**Step 7 – Architecture & Hyperparameters**
Go to the **Training page**:

- **Architecture:** Start with `ResNet-18`
- **Epochs:** 20–30 for initial tests, 50+ for final models
- **Device:** `auto` automatically uses GPU / MPS / CPU
- **Early Stopping:** 5–7 epochs prevents overfitting

**Step 8 – Start Training**
Click *Training starten*. You see Train-Loss, Val-Loss, and Accuracy updating in real time.

The best model (highest Val-Accuracy) is saved automatically.

> 💡 **How do I recognize good training?**
> ✓ Val-Loss decreases together with Train-Loss
> ✓ Val-Accuracy rises continuously
> ✗ Val-Loss rises while Train-Loss falls = overfitting

---

## Phase 5 – Deploy the Model

**Step 9 – Classify New Images**
Go to the **Classification (Inference) page**:
1. *Load model (.pth)* → select the model file
2. *Folder…* → select a folder with new images
3. *Classify all images* → results with Top-3 predictions

**Step 10 – Export Results**
Go to the **Export page**:
1. *Load results from last inference*
2. Configure columns (enable, rename)
3. *Export to Excel*

> 💡 **Done!** For better results: collect more data → re-label → repeat training.
