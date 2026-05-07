# Molecular Design VAE

A multi-stage generative framework for de novo drug discovery. Combines a SELFIES-based Beta-VAE with PPO reinforcement learning, NSGA-II Pareto ranking, and an interactive web interface for molecular analysis.

> M. Bhagya Sri (SR No: 27003) · Simarpreet Kaur (SR No: 26727)  
> Generative AI for Research — coursework project

---

## Overview

The pipeline learns a continuous latent representation of drug-like chemical space from a curated seed library, then uses reinforcement learning to navigate this space toward molecules with desirable properties: drug-likeness (QED), synthetic accessibility (SA), predicted binding affinity, and ADMET profile. SELFIES encoding guarantees that every generated structure is chemically valid by construction.

The system provides a complete interactive analysis platform — chemical space PCA, structure-activity relationship heatmaps, activity cliff detection, matched molecular pair analysis, scaffold hopping, retrosynthetic route suggestion, and 3D structure visualisation — all accessible through a browser interface.

---

## Try it

**Online (zero setup):**  
https://kaur-simarpreet-molecular-design-vae.hf.space

**Locally — fastest path (download pre-trained weights, no training):**
```bash
git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
cd molecular-design-vae
pip install -r requirements.txt
python download_assets.py                  # one-time, bundles JS for offline use
python download_weights.py --mode basic    # ~2 min, skips training entirely
python serve.py
# open index.html in your browser
```

**Locally — train from scratch:**
```bash
git clone https://github.com/Kaur-Simarpreet/molecular-design-vae.git
cd molecular-design-vae
pip install -r requirements.txt
python download_assets.py
python train_vae.py --mode basic           # 5-8 min on CPU
python serve.py
```

---

## Training modes

The training script supports four modes covering different dataset sizes and hardware:

| Mode | Dataset | Epochs | Hardware | Time |
|------|---------|--------|----------|------|
| `basic` | 295 curated seeds | 150 | CPU | 5–8 min |
| `moderate` | ~25k ZINC molecules | 250 | CPU | 60–90 min |
| `large` | ~100k ZINC molecules | 400 | CPU | 4–6 hrs |
| `large-gpu` | ~100k ZINC molecules | 400 | GPU | 20–40 min |

Detailed setup instructions for each mode are in `docs/`:
- [`SETUP_BASIC.md`](docs/SETUP_BASIC.md)
- [`SETUP_MODERATE.md`](docs/SETUP_MODERATE.md)
- [`SETUP_LARGE.md`](docs/SETUP_LARGE.md)
- [`SETUP_LARGE_GPU.md`](docs/SETUP_LARGE_GPU.md)

The deployed web app runs `basic` mode. To use other modes locally, you can either train from scratch or download pre-trained weights.

### Pre-trained weights

Trained model weights for all four modes are hosted on Hugging Face Hub:

**https://huggingface.co/Kaur-Simarpreet/molecular-design-vae**

Skip training entirely with:

```bash
python download_weights.py --mode basic       # ~6 MB,  2 min
python download_weights.py --mode moderate    # ~6 MB,  2 min
python download_weights.py --mode large       # ~20 MB, 3 min
python download_weights.py --mode large-gpu   # ~20 MB, 3 min  (same file as 'large')
```

Or run interactively:
```bash
python download_weights.py
```

The script downloads `vae.pt`, `tokenizer.pkl`, `latents.pt`, and `config.json` into `saved_model/` — exactly what `train_vae.py` would have produced. After downloading, run `python serve.py` and you're done.

For maintainers who want to publish their own trained weights, see `upload_weights.py`.

---

## Architecture

```
Seed library (ChEMBL 34 + ZINC-250K)
    │
    ▼  SELFIES tokenisation + RDKit validation
    │
    Beta-VAE
    ├── Encoder: Bidirectional LSTM → μ, log σ²
    ├── Latent:  z = μ + σ·ε  (reparameterisation)
    └── Decoder: Autoregressive LSTM → SELFIES tokens
    │
    ▼  Cyclic cosine KL annealing
    │
    PPO RL hill-climbing in latent space
    │  Reward = w_qed·QED + w_dock·dock + w_sa·(1-sa/10) + w_nov·novelty + w_admet·admet
    │
    ▼
    8-stage quality filter
    │  MW · Tanimoto diversity · Lipinski · QED min · PAINS · ADMET · Not-seed · Wet-lab
    │
    ▼
    NSGA-II Pareto ranking (QED↑, Docking↑, SA↓, Novelty↑)
    │
    ▼
    Wet-lab ready shortlist → Browser UI
```

### Hyperparameters by mode

| Parameter | basic | moderate | large / large-gpu |
|-----------|-------|----------|-------------------|
| Latent dim | 128 | 128 | 256 |
| LSTM hidden | 256 | 256 | 512 |
| Embedding dim | 64 | 64 | 128 |
| Max SELFIES length | 80 | 80 | 100 |
| Total epochs | 150 | 250 | 400 |
| Batch size | 64 | 128 | 256 / 512 |
| Learning rate | 5e-4 | 3e-4 | 2e-4 |
| β_max (KL) | 0.20 | 0.18 | 0.15 |
| Augmentation | 3× | 2× | 1× |

### Reward objectives

| Objective | QED | Docking | SA | Novelty | ADMET |
|-----------|-----|---------|-----|---------|-------|
| Binding affinity | 30% | 40% | 15% | 5% | 10% |
| Multi-objective | 30% | 25% | 15% | 15% | 15% |
| ADMET optimised | 30% | 15% | 10% | 5% | 40% |
| Novelty focused | 25% | 20% | 10% | 35% | 10% |
| Synthesisability | 25% | 15% | 45% | 5% | 10% |

---

## Dataset

Training data combines a hand-curated seed library with optional external datasets.

### Seed library (always included, 295 molecules)

Hand-picked from ChEMBL 34 and ZINC-250K across 12 scaffold families. All molecules pass Lipinski's Rule of Five and have molecular weight between 150 and 500 Da. RDKit validates each entry before inclusion.

| Scaffold family | Count |
|----------------|-------|
| Kinase inhibitor scaffolds | 28 |
| GPCR ligands | 18 |
| Protease inhibitors | 10 |
| Amino acid derivatives | 8 |
| Heterocyclics | 20 |
| Urea / hydroxamic acids | 15 |
| Sulfonamides | 13 |
| Fluorinated compounds | 10 |
| Morpholine / piperazine | 10 |
| Pyrimidine / triazine | 10 |
| Fused ring systems | 12 |
| Indole / benzimidazole | 8 |
| Named reference drugs | 16 |

Reference drugs include Aspirin, Ciprofloxacin, Celecoxib, Acetazolamide, Salbutamol, Adenosine, Nicotine, Paracetamol, and others.

### External data (moderate / large modes)

The training script downloads drug-like subsets from **ZINC-250K** — a publicly available benchmark dataset of 250,000 drug-like molecules from the ZINC database, filtered for oral bioavailability (MW 150–500 Da, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10). Downloaded data is cached locally so subsequent training runs do not require internet access.

---

## API

The Flask backend exposes the following endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Server status, model info, current mode |
| GET | `/modes` | Available training modes (loaded vs locked) |
| POST | `/generate` | Generate molecules for a target |
| POST | `/optimize` | Optimise a user-provided molecule |
| POST | `/scaffold_hop` | Scaffold hopping from a SMILES |
| POST | `/retro` | Retrosynthesis route suggestion |
| GET | `/sar` | SAR heatmap by functional group |
| GET | `/chemical_space` | Chemical space PCA coordinates |
| POST | `/mmp` | Matched molecular pair analysis |
| GET | `/export?fmt=csv\|sdf\|pdb` | Export results |

Example:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"target": "EGFR kinase", "objective": "binding_affinity", "num_mols": 6, "mode": "basic"}'
```

---

## Real docking (optional)

By default, docking scores are RDKit-based estimates. Real docking is available through three engines, in order of recommendation for novel molecules:

1. **Gnina** — CNN scoring, best for novel chemotypes (`docs/DOCKING_SETUP.md`)
2. **Vina** — classical force field, fast (`docs/DOCKING_SETUP.md`)
3. **DiffDock** — blind docking, no grid box (`docs/DIFFDOCK_SETUP.md`)

Default target: EGFR Kinase (PDB: 1IEP).

---

## Project structure

```
molecular-design-vae/
│
├── train_vae.py              Train the VAE — choose mode at runtime
├── serve.py                  Flask API server
├── index.html                Web UI
├── docking.py                Gnina + Vina docking integration
├── diffdock_cpu.py           DiffDock CPU wrapper
├── diffdock_gpu.py           DiffDock GPU wrapper
├── download_assets.py        Bundle JS/fonts for offline use
├── download_weights.py       Skip training — fetch pre-trained weights from HF Hub
├── upload_weights.py         Maintainer-only — publish weights to HF Hub
│
├── requirements.txt
├── requirements-docking.txt
├── Dockerfile                For Hugging Face Spaces deployment
│
├── saved_model/              Generated by train_vae.py
├── proteins/                 PDB receptor files (download separately)
├── receptors/                Auto-generated PDBQT receptors
├── static/                   Bundled JS (run download_assets.py first)
│
├── docs/
│   ├── SETUP_BASIC.md
│   ├── SETUP_MODERATE.md
│   ├── SETUP_LARGE.md
│   ├── SETUP_LARGE_GPU.md
│   ├── DOCKING_SETUP.md
│   ├── DIFFDOCK_SETUP.md
│   ├── HUGGINGFACE_DEPLOY.md
│   └── GITHUB_SETUP.md
│
├── COMPARISON.md             Comparison with related tools (separate document)
└── COMPARISON.tex            LaTeX version of comparison document
```

---

## Limitations

- **Default seed dataset is small** (295 molecules). Use moderate or large mode for broader chemical space coverage.
- **Default docking is estimated** (RDKit). Set up Gnina, Vina, or DiffDock for real binding scores.
- **No wet-lab validation** — all outputs are computational predictions.
- **Target-agnostic generation** — the VAE does not condition directly on protein pocket structure.
- **Approximately 50% quality filter pass rate** — SELFIES guarantees chemical validity, but only about half of generated molecules also pass all eight drug-like quality filters.

---

## Requirements

```
torch >= 1.12
selfies >= 2.1
rdkit >= 2022.03
flask >= 2.2
flask-cors >= 3.0
numpy >= 1.21
scipy >= 1.7
```

Optional for real docking:
```
meeko >= 0.4
vina >= 1.2
openbabel  # conda install -c conda-forge openbabel
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

- [SELFIES](https://github.com/aspuru-guzik-group/selfies) — Krenn et al. 2020
- [RDKit](https://www.rdkit.org/) — open-source cheminformatics
- [DiffDock](https://github.com/gcorso/DiffDock) — Corso et al. 2022
- [Gnina](https://github.com/gnina/gnina) — McNutt et al. 2021
- [AutoDock Vina](https://vina.scripps.edu/) — Eberhardt et al. 2021
- ZINC-250K — Irwin et al.
- ChEMBL 34 — Mendez et al.
- EGFR PDB 1IEP — used as primary docking target
