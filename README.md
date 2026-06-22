# End-to-End Deep Learning for Autonomous Driving

A project that trains neural networks to predict steering angles directly from front-facing camera images, progressing from a stateless CNN to a sequence-based C-LSTM, with an optional Behavior Transformer (BeT) extension.

---

## Overview

This project implements three model architectures for autonomous steering prediction using the Udacity self-driving car simulator dataset:

| Part | Architecture | Key Idea |
|------|-------------|----------|
| 1 | EDA & Data Prep | Understand and balance the dataset |
| 2 | CNN | Single-frame spatial feature extraction |
| 3 | C-LSTM | CNN + LSTM for temporal memory |
| 4 | Behavior Transformer (BeT) | Discrete action binning + classification |

**Papers covered:**
- *End to End Learning for Self-Driving Cars* (NVIDIA, 2016)
- *End-to-End Deep Learning for Steering Autonomous Vehicles Considering Temporal Dependencies* (2017)
- *Behavior Transformers: Cloning k modes with one stone* (NeurIPS 2022) — extra credit

---

## Dataset

The project uses two driving tracks from the Udacity simulator, stored in `udacity_data.zip`:

- `self_driving_car_dataset_make` — Lake track (simpler environment)
- `self_driving_car_dataset_jungle` — Jungle track (more complex, used for validation)

Each track's `driving_log.csv` contains:

| Column | Description |
|--------|-------------|
| `center` | Path to the center camera image |
| `left` | Path to the left camera image |
| `right` | Path to the right camera image |
| `steering` | Steering angle (0 = straight, + = right, − = left) |
| `throttle` | Throttle command (nearly always 1.0) |
| `brake` | Braking command (always 0) |
| `speed` | Vehicle speed |
| `track_source` | Which folder/track the frame came from |

**Total frames:** 7,334 across both tracks.

---

## Setup

### Requirements

```bash
pip install torch torchvision opencv-python scikit-learn pandas numpy matplotlib
```

### Data

Place `udacity_data.zip` in the project root. The notebook will automatically extract it to `./udacity_data/` on first run.

---

## Project Structure

```
├── udacity_data/
│   ├── self_driving_car_dataset_make/
│   │   ├── IMG/
│   │   └── driving_log.csv
│   └── self_driving_car_dataset_jungle/
│       ├── IMG/
│       └── driving_log.csv
├── notebook.ipynb       # Main project notebook
└── README.md
```

---

## Part 1: Exploratory Data Analysis

Key findings from EDA:

- **`brake`** is constant at 0.0 across all 7,334 frames — the vehicle never brakes.
- **`throttle`** is nearly constant at 1.0 (99.8% of frames) — always at full acceleration.
- **`speed`** is tightly clustered around 30 mph.
- **`steering`** is heavily right-skewed toward 0: **61.8% of frames are straight-driving** (angle = 0.0).

### Steering Imbalance Fix

A `flatten_distribution` function drops 95% of zero-steering frames at random, reducing the straight-driving bias and producing a more balanced training distribution (3,054 frames after balancing).

### Multi-Camera Recovery Augmentation

Left and right camera images simulate lane-drift recovery by applying a ±0.2 steering correction:

- **Left camera** → add +0.2 (steer right to recover to center)
- **Right camera** → subtract −0.2 (steer left to recover to center)

---

## Part 2: CNN Model

### Data Split

| Split | Source | Size |
|-------|--------|------|
| Train | Make track + jungle subset | 4,882 frames |
| Validation | Jungle track subset | 1,452 frames |
| Test | 500 frames each track | 1,000 frames |

### Architecture: `ConstrainedDrivingCNN`

```
Input (3 × 160 × 320)
  → Normalize (/255 − 0.5)
  → Crop (rows 70 to 136, removing sky and hood)
  → Conv2d(3→24, 5×5, stride 2) + ReLU
  → Conv2d(24→36, 5×5, stride 2) + ReLU + Dropout2d(0.10)
  → Conv2d(36→48, 5×5, stride 2) + ReLU
  → Conv2d(48→64, 3×3, stride 2) + ReLU
  → Conv2d(64→64, 3×3, stride 1, padding 1) + ReLU
  → Flatten (→ 2,304)
  → Linear(2304→100) + ReLU + Dropout(0.30)
  → Linear(100→50) + ReLU
  → Linear(50→1)   ← steering angle output
```

**Trainable parameters:** 366,949

### Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 16 |
| Epochs | 5 |
| Loss | MSE |
| Steering correction | ±0.2 |

### Data Augmentation (per epoch)

- **Dynamic filtering:** drop 50% of zero-steering frames each epoch
- **Recovery augmentation:** use left/right camera images with ±0.2 correction
- **Symmetry augmentation:** randomly flip images horizontally (mirror steering angle sign)

---

## Part 3: C-LSTM Model

### Motivation

The CNN is stateless — it sees only one frame at a time. Real driving is temporal: the car's recent history matters for smooth steering. The C-LSTM adds an LSTM layer on top of the CNN to capture this memory.

### Architecture: `CLSTMDrivingModel`

```
Input: sequence of 5 frames (5 × 3 × 160 × 320)

For each frame t in [1..5]:
  → Same CNN spatial encoder as Part 2
  → Feature vector (2,304-dim)

Sequence of 5 feature vectors
  → LSTM(input=2304, hidden=128, batch_first=True)
  → Take final hidden state

→ Linear(128→64) + ReLU + Dropout(0.30)
→ Linear(64→1)   ← steering angle output
```

**Trainable parameters:** 1,385,877

### Key Design Choices

- **Shared CNN weights** across all time steps (same spatial encoder)
- **Sequence length:** 5 frames (current + 4 preceding)
- **No shuffle** when building sequences — temporal order is preserved per track
- The LSTM's gating mechanism (input/forget/output gates) learns which historical frames are most relevant

### Results

| Model | Final Jungle Validation MSE |
|-------|-----------------------------|
| CNN | 0.107769 |
| C-LSTM | 0.085311 |

The C-LSTM reduces validation MSE by ~21% on the jungle track by incorporating temporal context.

---

## Part 4: Behavior Transformer (Extra Credit)

Based on the NeurIPS 2022 BeT paper, this part reformulates steering prediction as **discrete classification** rather than continuous regression.

### Action Discretization

K-means clustering (k=10) groups continuous steering angles into 10 discrete bins:

```
[-0.973, -0.725, -0.513, -0.310, -0.147,
  0.000,  0.246,  0.490,  0.735,  0.984]
```

### Architecture: `BeTCNN`

Same CNN feature extractor as Part 2, but the regression head is replaced with a **10-class classifier** (CrossEntropyLoss instead of MSE).

### Key Observation

Predictions appear **"steppy"** — the model can only output one of 10 fixed steering values, causing abrupt jumps between bins rather than smooth transitions. This is a direct consequence of discretizing the continuous action space without a residual correction.

The BeT paper addresses this with a **residual action head** that predicts a small continuous offset on top of the discrete bin center, enabling fine-grained steering adjustments within each bin. Without it, the model cannot represent the infinite variety of real steering angles.

---

## Hyperparameters Reference

```python
RANDOM_STATE     = 1746
BATCH_SIZE       = 16
EPOCHS           = 5        # CNN and C-LSTM
BET_EPOCHS       = 3        # BeT
LEARNING_RATE    = 0.001
STEERING_CORRECTION = 0.2   # left/right camera offset
SEQ_LENGTH       = 5        # C-LSTM sequence length
K                = 10       # BeT action bins
```

---

## Reproducibility

A global seed function is used throughout:

```python
def set_seed(seed: int = 1746):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

---

## References

1. Bojarski et al. (2016). *End to End Learning for Self-Driving Cars.* NVIDIA.
2. Xu et al. (2017). *End-to-End Deep Learning for Steering Autonomous Vehicles Considering Temporal Dependencies.*
3. Shafiullah et al. (2022). *Behavior Transformers: Cloning k modes with one stone.* NeurIPS.

---

*Made by orfegrace*
