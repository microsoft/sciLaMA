# sciLaMA

sciLaMA: A Single-Cell Representation Learning Framework to Leverage Prior Knowledge from Large Language Models

## Installation

This package uses `pyproject.toml` and is compatible with modern Python package managers like `uv`.

```bash
# Install with uv (recommended; creates uv.lock for reproducible installs)
uv sync

# Or with pip
pip install -e .
```


## Tutorial

### Quick start

1. **Prepare your data**: an `.h5ad` file with `adata.obs["split"]` containing `"train"`, `"val"`, and optionally `"test"`. Expression in `adata.X` should be normalized/scaled (mean≈0, std≈1).
2. **Edit** `script/template.yaml`: set `data.path` to your h5ad, and optionally add covariates or external feature embeddings.
3. **Train** (checkpoints and embeddings are saved automatically):

```python
from sciLaMA import SciLaMATrainer

trainer = SciLaMATrainer("script/template.yaml")
trainer.train()

# Access results (saved to save_dir during train())
adata = trainer.datamodule.adata
sample_emb = adata.obsm["X_sciLaMA"]  # (n_cells, latent_dim)
```

### Load from checkpoint

```python
trainer = SciLaMATrainer("script/template.yaml")
trainer.load_checkpoint("results/X_sciLaMA_direct.ckpt")
# Use trainer.module and trainer.datamodule for inference or analysis
```

### Training modes

```yaml
# script/template.yaml
training:
  mode: "direct"      # joint sample + feature VAE (needs external embeddings or uses Xt)
  # mode: "stepwise"  # phase 1: sample VAE; phase 2: feature VAE; phase 3: joint sample + feature VAE
  # mode: "beta_vae"  # sample (beta) VAE only (no feature VAE)
```

### External feature embeddings (multi-modal)

If you have precomputed gene embeddings (e.g. from a language model), add them to config. Parquet format: `gene_id`, `embedding` columns.

```yaml
data:
  external_feature_embeddings:
    gene_text: "data/gene_text.parquet"
    gene_protein: "data/gene_protein.parquet"
```

Genes are intersected across adata and all parquets; `adata.var["static_embedding"] = True` marks genes with embeddings.

---

### Step-by-step walkthrough

This tutorial alternates short explanations with runnable code blocks. Execute each in order.

**1. Prepare your AnnData.** Expression should be normalized/scaled; include a split column.

```python
import scanpy as sc

adata = sc.read_h5ad("data.h5ad")
# adata.obs["split"] should contain "train", "val", "test"
adata.write_h5ad("data_ready.h5ad")
```

**2. (Optional) Build external gene embeddings** as Parquet with `gene_id` and `embedding` columns.

```python
import pandas as pd
import numpy as np

genes = adata.var_names.tolist()
embeddings = [np.random.randn(128).tolist() for _ in genes]
pd.DataFrame({"gene_id": genes, "embedding": embeddings}).to_parquet("gene_text.parquet")
```

**3. Create a config** (YAML file or dict).

```python
import yaml

config = {
    "data": {"path": "data_ready.h5ad", "split_column": "split", "external_feature_embeddings": {"gene_text": "gene_text.parquet"}, "check_scaling": True},
    "model": {"hidden_dims": [256, 128], "latent_dim": 32},
    "training": {"mode": "direct", "max_epochs": 50, "batch_size": 64},
    "output": {"save_dir": "./results", "save_key": "X_sciLaMA"},
}
with open("my_config.yaml", "w") as f:
    yaml.dump(config, f)
```

**4. Train** (outputs saved automatically).

```python
from sciLaMA import SciLaMATrainer

trainer = SciLaMATrainer("my_config.yaml")
trainer.train()
```

**5. Inspect outputs** (saved to save_dir).

```python
adata = trainer.datamodule.adata
print("Sample embeddings:", adata.obsm["X_sciLaMA"].shape)

import pandas as pd
feat = pd.read_parquet("results/feature_embeddings_direct_sciLaMA.parquet")  # or stepwise_sciLaMA, etc.
print("Feature embeddings:", feat.shape)
```

## Configuration

`script/template.yaml` controls data, model, training, and output.


### Input: external feature embeddings (multi-modal)

`external_feature_embeddings` is a dict mapping **modality name → path** to Parquet files:

```yaml
external_feature_embeddings:
  gene_text: "data/gene_text.parquet"        # modality 1: NCBI gene card -> text-embedding model
  gene_protein: "data/gene_protien.parquet"  # modality 2: protein sequence -> ESM2 embedding 
```

**Per-file format:**

- **gene_id**: gene ID (required; must match `adata.var_names`)
- **embedding**: required column where each entry is a vector (list/array of floats)
- Each file can have different embedding lengths per modality

**Multi-file handling:**
- Genes are **intersected** across adata and all parquets; adata is subset to the common set
- Intersection counts are printed (per source and final)
- `adata.var["static_embedding"] = True` is set for genes in the intersection
- Each modality’s embeddings are aligned to the same gene order
- The MultiModalFeatureEncoder fuses these into one latent per gene (average, MoE, or PoE)

### Output: saved files

After training, the following are saved automatically:

- **Checkpoints**: `{save_key}_{mode}.ckpt` in `save_dir`
- **Sample embeddings**: `adata.obsm[save_key]` and mode-specific Parquet (e.g. `sample_embeddings_beta_vae.parquet`, `sample_embeddings_direct_sciLaMA.parquet`, `sample_embeddings_stepwise_sciLaMA.parquet` after stepwise phase 3)
- **Gene embeddings** (when model has feature VAE): mode-specific Parquet (e.g. `feature_embeddings_direct_sciLaMA.parquet`, `feature_embeddings_intermediate.parquet` after stepwise phase 2, `feature_embeddings_stepwise_sciLaMA.parquet` after stepwise phase 3)




## Citation

```bibtex
@inproceedings{hu2025scilama,
  title={sciLaMA: A Single-Cell Representation Learning Framework to Leverage Prior Knowledge from Large Language Models},
  author={Hu, H. and Zhang, S. and Choi, Y. and Malladi, V. S. and Quon, G.},
  booktitle={42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```

Hu, H., Zhang, S., Choi, Y., Malladi, V. S., & Quon, G. (2025). sciLaMA: A Single-Cell Representation Learning Framework to Leverage Prior Knowledge from Large Language Models. In *Forty-second International Conference on Machine Learning (ICML)*.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
