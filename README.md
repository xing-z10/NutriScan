# NutriScan: Does Depth Help Nutrition Estimation Generalize to Unseen Foods?

CS7150 Deep Learning term project — Na Yin, Xing Zhou

From a single overhead photo of a dish, we predict five values — **calories, mass, fat, carbs, protein** — and test whether adding depth helps a model generalize to **food types it never saw in training**.

Full proposal: [`Proposal_Depth_Unseen_Foods - Na & Xing.pdf`](Proposal_Depth_Unseen_Foods%20-%20Na%20&%20Xing.pdf)

## Research Questions

- **RQ1:** Does depth help more on unseen foods than on seen foods?
- **RQ2:** On unseen foods, does the error come from portion size or from food type?
- **RQ3 (optional):** Can depth estimated from a normal photo replace a depth camera?

## Approach

**Data.** [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) — real cafeteria dishes with an overhead RGB photo, a depth image, and weighed nutrition labels. 3,490 of 5,006 dishes have overhead RGB-D.

**Seen vs. unseen split.** Hold out one food category (e.g. fruit) and split those dishes into two halves. Both settings are tested on the same Half 2:

| Setting | Training set |
|---|---|
| Unseen | other dishes only (held-out category never appears) |
| Seen | fewer other dishes + Half 1 (same size as Unseen) |

Repeated scans of the same plate are grouped (`plate_group`) and kept on the same side of the split to avoid leakage.

**Models** (shared ImageNet-pretrained ResNet-50 backbone):

| Model | Input | Phase |
|---|---|---|
| A | RGB photo | 1 |
| B | RGB + camera depth | 1 |
| RGB-D fusion network | RGB + camera depth | 1 (extension) |
| C | RGB + estimated depth (Depth Anything V2) | 2 (optional) |

**Metric.** Mean absolute error (absolute and %) with confidence intervals. The key quantity is the **gap** = unseen error − seen error; depth helps if model B's gap is clearly smaller than model A's. For RQ2, mass error (portion) and calories-per-gram error (food type) are evaluated separately.

## Repository Structure

```
.
├── DataProcess.ipynb          # Data audit: parse metadata, food categories, plate grouping
├── ingredient_summary.csv     # Per-ingredient stats (dishes, total grams, dishes with ≥30% share)
├── dish_manifest_draft.csv    # Per-dish labels, category fractions, plate_group (draft)
├── requirements.txt
└── Proposal_Depth_Unseen_Foods - Na & Xing.pdf
```

### `dish_manifest_draft.csv` columns

| Column | Meaning |
|---|---|
| `dish_id` | Nutrition5k dish id |
| `cafe` | `cafe1` / `cafe2` |
| `cal`, `mass`, `fat`, `carb`, `protein` | Labels (kcal, g, g, g, g) |
| `n_ingr` | Number of ingredients |
| `has_depth` | Has overhead RGB-D (all `True` after filtering) |
| `frac_seafood`, `frac_chicken`, `frac_egg`, `frac_fruit` | Mass fraction of the dish belonging to each category |
| `ts` | Timestamp parsed from `dish_id` |
| `plate_group` | Reconstructed group of incremental scans of the same plate (heuristic) |

## Getting Started

The notebook is designed for Google Colab.

```bash
pip install -r requirements.txt
```

Download the metadata (a few MB):

```bash
mkdir -p n5k/metadata
gsutil -m cp "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/*" n5k/metadata/
gsutil ls gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/ > n5k/overhead_dirs.txt
```

Then run `DataProcess.ipynb`. The raw dataset (`n5k/`) is not tracked in git.

## Status

- [x] Proposal
- [x] Data audit: metadata parsing, food categories, plate grouping
- [ ] Final seen/unseen split
- [ ] Image + depth download and data loader
- [ ] Phase 1: models A, B, RGB-D fusion
- [ ] Evaluation and error analysis
- [ ] Phase 2 (optional): estimated depth
- [ ] Final report

## References

1. Q. Thames et al., "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food," CVPR 2021.
2. "RDINet: A Deep Learning Model Integrating RGB-D and Ingredient Features for Food Nutrition Estimation," Applied Sciences, 2026.
3. Y. Han et al., "DPF-Nutrition: Food Nutrition Estimation via Depth Prediction and Fusion," Foods, 2023.
4. CalorieLLaVA, a fine-tuned LLaVA model for calorie estimation, 2024.
5. L. Yang et al., "Depth Anything V2," NeurIPS 2024.
