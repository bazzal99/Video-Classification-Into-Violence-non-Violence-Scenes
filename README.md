# Video Violence Detection

Binary video classification into **Violence** and **Non-Violence** using a
**ConvLSTM**-based deep learning model trained on the
[Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
(2000 videos). Includes a Flask web interface for real-time video upload and prediction.

> Final model accuracy: **93%** — F1-score 0.93 (balanced across both classes)

---

## Pipeline overview

```
Real Life Violence Dataset (Kaggle)
        ↓
1_preprocess.ipynb
  → Uniform frame extraction (60 frames/video)
  → Resize to 64×64, normalise to [0,1]
  → Save as dataset.npz  (N, 60, 64, 64, 3)
        ↓
2_train.ipynb
  → Temporal augmentation (video reversal → 2× dataset)
  → ConvLSTM2D × 2  +  Dense classifier
  → SGD (lr=1e-3), EarlyStopping
  → Save model.keras
        ↓
3_deploy.ipynb  /  app.py
  → Flask backend + localtunnel
  → Web UI: upload video → Violence / Non-Violence + confidence
```

---

## Results

| Experiment | Dataset | Architecture | Accuracy |
|------------|---------|-------------|----------|
| 1 | Original (1996 videos) | 1× ConvLSTM | 88% |
| 2 | Augmented (3992 videos) | 1× ConvLSTM | 91% |
| 3 ✅ | Augmented (3992 videos) | 2× ConvLSTM (stacked) | **93%** |

Best model classification report:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| NonViolence | 0.92 | 0.96 | 0.94 |
| Violence | 0.95 | 0.91 | 0.93 |

---

## Setup

```bash
git clone https://github.com/bazzal99/Video-Classification-Into-Violence-non-Violence-Scenes
cd Video-Classification-Into-Violence-non-Violence-Scenes
pip install -r requirements.txt
```

---

## Usage

### 1. Preprocess the dataset (Kaggle)

Upload `1_preprocess.ipynb` to Kaggle, attach the
[Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset),
and run all cells. This produces `dataset.npz`.

### 2. Train the model (Kaggle)

Upload `2_train.ipynb` and `dataset.npz` to Kaggle, then run all cells.
This produces `model.keras`. Experiment 3 (last section) is the best configuration.

### 3. Deploy the web interface (Colab or local)

**Colab:**
Upload `app.py`, `model.keras`, and `static/index.html` to Colab,
then run `3_deploy.ipynb` to get a public URL via localtunnel.

**Local:**
```bash
python app.py
# Open http://localhost:5000
```

---

## Files

| File | Description |
|------|-------------|
| `1_preprocess.ipynb` | Frame extraction and dataset creation |
| `2_train.ipynb` | Model training — all 3 experiments with clear sections |
| `3_deploy.ipynb` | Flask deployment via localtunnel (Colab) |
| `app.py` | Flask backend — handles video upload, preprocessing, prediction |
| `static/index.html` | Web UI — drag-and-drop video upload with confidence display |
| `requirements.txt` | Python dependencies |

> **Note:** `model.keras` and `dataset.npz` are excluded from the repo (too large).
> Train the model yourself using the notebooks above.

---

## Dataset

[Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) — Mohamed Mustafa, Kaggle, 2020.
2000 videos (1000 Violence + 1000 NonViolence), real-world scenarios, avg. ~7 seconds at 30 fps.

---


## License

MIT License. See [LICENSE](LICENSE).
