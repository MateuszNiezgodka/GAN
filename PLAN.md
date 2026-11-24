# EMNIST Letters GAN — Project Implementation Plan

> **Course:** Deep Learning Labs (PUT)  
> **Team Size:** 2 students (remember to include both student IDs in the final submission; only one person submits)  
> **Dataset:** EMNIST Letters (`28x28` grayscale images, 26 letter classes `A–Z` merged across upper/lowercase)  
> **Important Deadline:** If aiming for **Grade 5.0**, at least one consultation (online or in person) is required **by December 15th** — email `gmiebs@cs.put.poznan.pl` with subject starting with `[DL]`.

---

## 1. Guiding Philosophy (Keep It Simple & Incremental)

Because we are not sure yet whether we will stop at **3.0**, **4.0**, or **5.0**, the project is designed so that **each phase directly reuses the code and architecture from the previous phase**:

```
Phase 1 (Grade 3.0)          Phase 2 (Grade 4.0)               Phase 3 (Grade 5.0)
Unconditional DCGAN    ───►  Conditional DCGAN (cGAN)    ───►  One-Shot Feature-Conditioned GAN
G(z), D(x)                   G(z, e_c), D(x, e_c)              G(z, E(x_ref)), D(x, E(x_ref))
                             where e_c = Embedding(c)          where E is a CNN Encoder on 1 example
```

Notice the key insight connecting Phase 2 and Phase 3:
- In **Phase 2 (Grade 4.0)**, we condition the Generator and Discriminator on a 32-dim vector $e_c$ from a lookup table `nn.Embedding(26, 32)`.
- In **Phase 3 (Grade 5.0)**, we simply replace that discrete lookup table with a small CNN Encoder $E(x_{\text{ref}}) \in \mathbb{R}^{32}$ that extracts the character shape embedding from a single reference image!

---

## 2. Repository Structure & Collaboration Workflow

Jupyter notebooks are great for showing visuals and narrating "trials and errors" (required by the lab instructions), but two people editing one huge `.ipynb` file on GitHub causes painful merge conflicts. We will use a **lightweight hybrid structure**:

```text
GAN/
├── README.md                  # Lab task description
├── PLAN.md                    # This implementation plan
├── pyproject.toml             # Dependencies (torch, torchvision, matplotlib, tqdm, jupyter)
├── src/
│   ├── __init__.py
│   ├── dataset.py             # EMNIST loader, orientation fix, class split for Grade 5.0
│   ├── models.py              # Generator, Discriminator, and Grade 5.0 Shape Encoder
│   └── utils.py               # Image grid plotting, checkpoint saving, seed setting
└── notebooks/
    ├── 01_grade3_basic_gan.ipynb      # Phase 1: Unconditional GAN + trials & errors report
    ├── 02_grade4_conditional_gan.ipynb# Phase 2: Conditional GAN + interactive letter generation
    └── 03_grade5_oneshot_gan.ipynb    # Phase 3: Unseen letter generation from 1 example
```

### Environment Setup Note
- `.python-version` is currently set to `3.14`. If PyTorch (`torch`, `torchvision`) prebuilt wheels are not yet available for Python 3.14 on your machine, switch `.python-version` and `pyproject.toml` to Python `3.12` (`uv python pin 3.12`), or run training notebooks on **Google Colab / Kaggle (free T4 GPU)** while keeping code synced via GitHub.
- Core dependencies to install: `torch`, `torchvision`, `matplotlib`, `numpy`, `tqdm`, `ipykernel`.

---

## 3. Shared Data Pipeline & Two Critical EMNIST Pitfalls (`src/dataset.py`)

Before building any GAN, implement and test the data loader first. There are **two famous bugs** with `torchvision.datasets.EMNIST(split="letters")` that trip up almost every student group:

1. **Pitfall #1 — Rotated & Flipped Images:**
   - By default, `torchvision` loads EMNIST images transposed (rotated 90° and flipped horizontally) because of the original MATLAB file layout.
   - **Fix:** Add `transforms.Lambda(lambda x: x.transpose(1, 2))` right after `transforms.ToTensor()` so all letters appear upright and human-readable.
2. **Pitfall #2 — 1-Indexed Labels (`1..26` instead of `0..25`):**
   - In `EMNIST(split="letters")`, class `0` is unused ("N/A") and letters `'A'..'Z'` have labels `1..26`.
   - **Fix:** Use `target_transform=lambda y: y - 1` in the dataset constructor so labels are `0..25` (`0 = 'A'`, ..., `25 = 'Z'`).
3. **Pixel Normalization for GANs:**
   - Normalize images from `[0, 1]` to `[-1, 1]` with `transforms.Normalize((0.5,), (0.5,))` to match the Generator's final `nn.Tanh()` activation.

**Verification Check:** Plot a `4x8` grid of real training images with their decoded letter titles (`chr(label + ord('A'))`) to confirm letters are upright and labels match before writing any GAN code.

---

## 4. Phase 1 — Grade 3.0: Basic Unconditional GAN

### Objective
Train a GAN that maps random Gaussian noise $z \sim \mathcal{N}(0, I_{64})$ to realistic `1x28x28` handwritten letter images.

### Step-by-Step Implementation
1. **Baseline "Trial & Error" Model (MLP GAN):**
   - *Why build this first?* The assignment explicitly requires showing **trials and errors** in the notebook/report.
   - Build a tiny 3-layer `nn.Linear` Generator and Discriminator (`LeakyReLU(0.2)`, `Dropout(0.3)`).
   - Train for 15–20 epochs and save sample grids. Note in the report that MLP GANs struggle with spatial coherence and produce grainy backgrounds.
2. **Main Phase 1 Model (Compact DCGAN for `28x28`):**
   - **Generator $G(z)$:**
     - Input: $z \in \mathbb{R}^{64}$
     - `Linear(64, 128 * 7 * 7)` $\rightarrow$ reshape to `(128, 7, 7)` $\rightarrow$ `BatchNorm2d(128)` + `ReLU`
     - `ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)` $\rightarrow$ `(64, 14, 14)` $\rightarrow$ `BatchNorm2d(64)` + `ReLU`
     - `ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)` $\rightarrow$ `(1, 28, 28)` $\rightarrow$ `Tanh()`
   - **Discriminator $D(x)$:**
     - Input: $x \in \mathbb{R}^{1 \times 28 \times 28}$
     - `Conv2d(1, 64, kernel_size=4, stride=2, padding=1)` $\rightarrow$ `(64, 14, 14)` $\rightarrow$ `LeakyReLU(0.2)`
     - `Conv2d(64, 128, kernel_size=4, stride=2, padding=1)` $\rightarrow$ `(128, 7, 7)` $\rightarrow$ `BatchNorm2d(128)` + `LeakyReLU(0.2)`
     - Flatten $\rightarrow$ `Linear(128 * 7 * 7, 1)` (output raw logits for `nn.BCEWithLogitsLoss()`)
3. **Training Recipe & Stability Tricks:**
   - Optimizer: `Adam(lr=2e-4, betas=(0.5, 0.999))` for both $G$ and $D$.
   - Batch size: `128`, Epochs: `25–30` (~5–8 minutes on a single GPU).
   - **One-sided label smoothing:** Use `0.9` instead of `1.0` for real image labels in $D$'s loss so $D$ does not become overconfident and kill gradients to $G$.
   - Track fixed-noise progression: Evaluate $G(z_{\text{fixed}})$ after every epoch and save comparison grids at Epoch 1, 5, 15, 30.

### Phase 1 Definition of Done (Ready to Submit for 3.0)
- Notebook `01_grade3_basic_gan.ipynb` shows:
  - Dataset visualization (with orientation fix explained).
  - Trial 1 (MLP GAN) vs. Trial 2 (DCGAN) comparison and discussion of training dynamics ($G$ vs. $D$ loss curves).
  - Final grid of sharp, recognizable generated handwritten letters.

---

## 5. Phase 2 — Grade 4.0: Conditional GAN (User-Specified Letter)

### Objective
Allow the user to specify any letter `'A'..'Z'` and generate multiple diverse handwritten images of that exact letter.

### Step-by-Step Implementation
1. **Architecture Upgrade (cDCGAN):**
   - Add a learnable class embedding layer: `nn.Embedding(num_classes=26, embedding_dim=32)`.
   - **Conditional Generator $G(z, c)$:**
     - Look up $e_c = \text{Embedding}_G(c) \in \mathbb{R}^{32}$.
     - Concatenate $[z, e_c] \in \mathbb{R}^{64 + 32 = 96}$.
     - Pass into the same DCGAN Generator backbone (`Linear(96, 128 * 7 * 7)` $\rightarrow$ `ConvTranspose2d` blocks).
   - **Conditional Discriminator $D(x, c)$:**
     - Look up $e_c = \text{Embedding}_D(c)$ and project via `Linear(32, 28 * 28)` $\rightarrow$ reshape to `(1, 28, 28)`.
     - Concatenate with the image along the channel dimension to get a `(2, 28, 28)` tensor, then feed into `Conv2d(2, 64, 4, 2, 1)` and the rest of the Phase 1 Discriminator.
     - *(Alternative to document in Trials & Errors: concatenating $e_c$ at the flattened `7x7` feature layer or adding an Auxiliary Classifier head — AC-GAN).*
2. **User Generation Interface:**
   - Write a clean helper function in the notebook:
     ```python
     def generate_letter(letter: str, num_samples: int = 8, model=G):
         """Generate `num_samples` variations of a user-specified letter ('A'-'Z')."""
     ```
   - Also bonus demo: a `generate_word(word: str)` function that calls `generate_letter` for each character in a word (e.g., `"PUT"`, `"DEEP"`) and stitches them side-by-side!

### Phase 2 Definition of Done (Ready to Submit for 4.0)
- Notebook `02_grade4_conditional_gan.ipynb` shows:
  - Full alphabet matrix (`26 rows x 8 columns`) showing 8 generated handwriting styles for every letter `A–Z`.
  - Interactive function where specifying a single letter (or word) outputs clean images of that letter.
  - Discussion of conditioning trials (e.g., embedding size, label smoothing, or early conditioning vs. late conditioning).

---

## 6. Phase 3 — Grade 5.0: One-Shot Generation of Previously Unseen Letters

### Objective
Train the GAN on a subset of the alphabet (e.g., **21 seen letters `A–U`**), completely holding out **5 unseen letters (`V, W, X, Y, Z`)**. At generation time, given **only ONE example image** $x_{\text{ref}}$ of an unseen letter (e.g., `'Z'`), generate diverse new handwritten images of that letter.

> **Reminder:** Schedule the mandatory consultation with `gmiebs@cs.put.poznan.pl` (subject `[DL] ...`) **before December 15th** to review this Phase 3 design!

### Why Phase 2 Cannot Do This Directly
In Phase 2, conditioning uses `nn.Embedding(26, 32)`. If letter `'Z'` is excluded from training, its embedding vector never receives gradient updates. To generalize to an unseen letter from a single image $x_{\text{ref}}$, we must extract the condition vector directly from the image using a **Shape Encoder $E(x_{\text{ref}}) \in \mathbb{R}^{32}$**.

### The Main Pitfall in One-Shot GANs (and How We Solve It)
If you naively train $G(z, E(x))$ to reconstruct the same image $x$, the Generator learns to **copy $x$ and ignore the noise $z$** (producing identical clones instead of diverse new samples). Conversely, if $D$ only checks "is the image a real letter?", $G$ may ignore $E(x_{\text{ref}})$ and generate random seen letters.

We solve both problems with **Cross-Example Pairing + Pre-trained Shape Encoder**:

### Step-by-Step Implementation
1. **Strict Class Split (`src/dataset.py`):**
   - **Seen classes ($\mathcal{C}_{\text{seen}}$):** Letters `A–U` (classes `0..20`, 21 classes) — used for training.
   - **Unseen classes ($\mathcal{C}_{\text{unseen}}$):** Letters `V–Z` (classes `21..25`, 5 classes) — strictly held out until evaluation!
2. **Stage 1 — Train the Shape Encoder $E(x)$ on $\mathcal{C}_{\text{seen}}$:**
   - Build a small 3-layer CNN Encoder $E(x)$ that outputs an L2-normalized embedding $e \in \mathbb{R}^{32}$, followed by a temporary classification head `Linear(32, 21)` (or supervised contrastive / triplet loss) trained for ~10 epochs on $\mathcal{C}_{\text{seen}}$ with random affine augmentation (small rotations $\pm 15^\circ$, scaling).
   - *Why this works:* Different letters share geometric strokes (lines, loops, corners, intersections). By learning to group different people's handwritings of the same seen letter together, $E(x)$ learns a **handwriting-invariant character shape descriptor** that generalizes zero-shot to `V–Z`!
   - *Sanity check before training the GAN:* Compute embeddings $E(x)$ on test images of unseen letters `V–Z` and verify with cosine similarity / t-SNE that unseen letters cluster cleanly by character class.
3. **Stage 2 — Train the One-Shot Conditional GAN $G(z, E(x_{\text{ref}}))$ on $\mathcal{C}_{\text{seen}}$:**
   - Freeze (or fine-tune with a $10\times$ smaller learning rate) the Shape Encoder $E$.
   - **Cross-Example Paired Sampling:** For each training image $x_{\text{real}}$ of class $c \in \mathcal{C}_{\text{seen}}$, sample a **different random image $x_{\text{ref}}$ from the same class $c$**.
   - Extract the condition vector $e_{\text{ref}} = E(x_{\text{ref}}) \in \mathbb{R}^{32}$.
   - **Generator $G(z, e_{\text{ref}})$:** Exact same architecture as Phase 2, taking $[z, e_{\text{ref}}] \in \mathbb{R}^{64 + 32}$!
     - *(Optional U-Net / Spatial skip variant for trials & errors: if a 32-dim bottleneck loses too much structure on complex unseen glyphs, use a light spatial bottleneck `(16, 7, 7)` with heavy dropout/instance normalization + AdaIN noise injection).*
   - **Discriminator $D(x, e_{\text{ref}})$:** Takes $(x_{\text{real}}, e_{\text{ref}})$ as real pairs and $(G(z, e_{\text{ref}}), e_{\text{ref}})$ as fake pairs. Also pass mismatched pairs $(x_{\text{real}}, e_{\text{wrong\_class}})$ as fake so $D$ enforces that the generated image matches the reference shape!
   - **Shape Consistency & Diversity Losses:**
     - **Content/Shape Loss:** $\mathcal{L}_{\text{shape}} = 1 - \cos\big(E(G(z, e_{\text{ref}})),\, e_{\text{ref}}\big)$ ensures the generated letter preserves the character identity of $x_{\text{ref}}$.
     - **Mode-Seeking Diversity Loss:** For two noise vectors $z_1, z_2$, maximize $\frac{\|G(z_1, e_{\text{ref}}) - G(z_2, e_{\text{ref}})\|_1}{\|z_1 - z_2\|_1}$ (clamped) so the Generator produces genuinely diverse handwriting styles instead of collapsing to a single image.
4. **One-Shot Evaluation on Unseen Letters (`V, W, X, Y, Z`):**
   - For each unseen letter in `{V, W, X, Y, Z}`, pick **1 single image** $x_{\text{ref}}$ from the test set.
   - Sample 10 random noise vectors $z_1, \dots, z_{10}$ and plot $x_{\text{ref}}$ next to $G(z_i, E(x_{\text{ref}}))$.
   - Quantitative evaluation (great for 5.0 report!): Train an oracle 26-class letter classifier on the real dataset and measure classification accuracy on the generated unseen letters + pixel diversity across different $z$.

---

## 7. Division of Work (2-Student Team)

| Milestone | Student A | Student B | Target |
| :--- | :--- | :--- | :--- |
| **Setup & Data** | Repo setup, `src/dataset.py` (orientation + label fix) | Data exploration plots, `src/utils.py` grid visualizer | Day 1 |
| **Phase 1 (Grade 3.0)** | Trial 1: MLP GAN baseline & loss logging | Trial 2: DCGAN (`src/models.py`) + label smoothing experiments | Week 1 |
| **Phase 2 (Grade 4.0)** | Conditional Generator & Discriminator (`cDCGAN`) | `generate_letter()` / `generate_word()` UI + alphabet grid plots | Week 2 |
| **Phase 3 (Grade 5.0)** | Shape Encoder $E(x)$ + unseen-class split (`A-U` vs `V-Z`) | Paired One-Shot GAN training loop + shape/diversity losses | Week 3 |
| **Report & Polish** | Compile trials & errors narrative + loss curves | Final visual comparisons + code cleanup | Before deadline |

---

## 8. Checklist for the Final Submission Report / Notebooks

- [ ] **Student IDs** of both team members clearly written at the top of the notebook / report.
- [ ] **Concept explanation:** Markdown sections explaining *why* each architecture and loss function was chosen.
- [ ] **Trials and errors documented:** Keep screenshots/plots of failed or suboptimal attempts (e.g., un-transposed EMNIST, MLP GAN graininess, mode collapse before label smoothing, naive one-shot identity copying before cross-example pairing) — the grading rubric specifically asks for this!
- [ ] **Consultation completed (if doing Grade 5.0):** Email sent to `gmiebs@cs.put.poznan.pl` with `[DL]` prefix and meeting held before **December 15th**.
