import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import os
from typing import Dict, Tuple, List, Any, TYPE_CHECKING


def is_rank_zero() -> bool:
    """True if current process is rank 0 (main process). Safe for single-GPU and multi-GPU."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return int(rank) == 0


def r0_print(*args: Any, **kwargs: Any) -> None:
    """Print only on rank 0. Use for all user-facing messages in multi-GPU."""
    if is_rank_zero():
        print(*args, **kwargs)


def r0_rich(*objects: Any, **kwargs: Any) -> None:
    """Rich-format output only on rank 0. Use for key info (phases, checkpoints, config summary)."""
    if not is_rank_zero():
        return
    try:
        from rich.console import Console
        Console().print(*objects, **kwargs)
    except ImportError:
        print(*objects, **kwargs)


def init_weights(m: nn.Module) -> None:
    """Xavier init for Linear layers. Use with module.apply(init_weights)."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

if TYPE_CHECKING:
    from .config import DataConfig

# Type for covariate encoding state saved in checkpoint: one-hot mapping for discrete, mean/std for continuous.
CovariateEncodingState = Dict[str, Any]  # {"discrete": {col: [cat1, ...]}, "continuous": {col: {"mean": float, "std": float}}}


def _categorical_covariate_columns(config: "DataConfig") -> List[str]:
    return list(config.categorical_covariate_keys) if config.categorical_covariate_keys else []


def fit_covariate_encoding_state(obs: pd.DataFrame, config: "DataConfig") -> CovariateEncodingState:
    """
    Fit encoding state from (e.g. train) obs. Use train-only obs so val/test and inference use same mapping/scale.
    Returns state to be saved in checkpoint and passed to encode_covariates.
    """
    state: CovariateEncodingState = {"discrete": {}, "continuous": {}}
    cat_cols = _categorical_covariate_columns(config)
    for col in cat_cols:
        if col not in obs.columns:
            continue
        cats = sorted(obs[col].astype(str).unique().tolist())
        state["discrete"][col] = cats
    cont_cols = config.continuous_covariate_keys or []
    for col in cont_cols:
        if col not in obs.columns:
            continue
        x = obs[col].astype(np.float64).values
        state["continuous"][col] = {"mean": float(np.nanmean(x)), "std": float(np.nanstd(x)) or 1.0}
    return state


def encode_covariates(obs: pd.DataFrame, encoding_state: CovariateEncodingState, config: "DataConfig") -> np.ndarray:
    """
    Encode obs to covariate matrix C: one-hot for discrete (using state["discrete"]), z-score for continuous (using state["continuous"]).
    Unknown categories at inference get all zeros for that one-hot block.
    """
    parts = []
    for col, categories in encoding_state.get("discrete", {}).items():
        if col not in obs.columns:
            n = len(obs)
            parts.append(np.zeros((n, len(categories)), dtype=np.float32))
            continue
        col_vals = obs[col].astype(str)
        onehot = np.zeros((len(obs), len(categories)), dtype=np.float32)
        for j, cat in enumerate(categories):
            onehot[:, j] = (col_vals == cat).astype(np.float32)
        parts.append(onehot)
    for col, scale in encoding_state.get("continuous", {}).items():
        if col not in obs.columns:
            parts.append(np.zeros((len(obs), 1), dtype=np.float32))
            continue
        x = obs[col].astype(np.float64).values
        mean, std = scale["mean"], scale["std"]
        parts.append(((x - mean) / (std or 1.0)).astype(np.float32).reshape(-1, 1))
    if not parts:
        return np.zeros((len(obs), 0), dtype=np.float32)
    return np.hstack(parts)


def _load_embedding_table(path: str) -> pd.DataFrame:
    """
    Load a gene embedding table from Parquet.
    **Expected format**: gene_id, embedding (both required columns).
    Each row: gene_name, embedding=[v1, v2, ...].
    Returns DataFrame with index=gene_names, values=embedding matrix (n_genes x dim).
    """
    if not path.lower().endswith(".parquet"):
        raise ValueError(
            f"Unsupported embedding file format: {path}. Only .parquet is supported."
        )
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise ValueError(f"Could not read Parquet file {path}: {e}") from e

    required = {"embedding", "gene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet must have 'embedding' and 'gene_id' columns. Missing: {missing}. Found: {list(df.columns)}")

    genes = df["gene_id"].astype(str).values

    emb_list = df["embedding"].tolist()
    emb_array = np.vstack([np.asarray(e, dtype=np.float32).flatten() for e in emb_list])
    return pd.DataFrame(emb_array, index=genes)


def check_normalized_scaled(adata: sc.AnnData, layer: str = None) -> bool:
    """
    Check if the data in adata.X (or layer) is normalized and scaled.
    Expects mean ~ 0 and std ~ 1.
    """
    X = adata.X if layer is None else adata.layers[layer]

    if hasattr(X, "toarray"):
        X = X.toarray()

    mean = np.mean(X)
    std = np.std(X)

    is_centered = np.abs(mean) < 0.1
    is_scaled = np.abs(std - 1.0) < 0.2

    if not (is_centered and is_scaled):
        r0_print(f"Warning: Data does not appear to be scaled. Mean: {mean:.4f}, Std: {std:.4f}")
        return False
    return True


def load_and_match_feature_embeddings(
    adata: sc.AnnData,
    embedding_paths: Dict[str, str]
) -> Tuple[sc.AnnData, Dict[str, torch.Tensor]]:
    """
    Load external feature embeddings from Parquet and match with adata.var_names.
    """
    if not embedding_paths:
        adata.var["static_embedding"] = True  # no external embeddings -> model all genes
        return adata, {}

    adata_genes = set(adata.var_names.astype(str))
    common_genes = adata_genes.copy()
    embeddings = {}
    file_gene_counts = {}

    for name, path in embedding_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")

        df = _load_embedding_table(path)
        df.index = df.index.astype(str)
        file_genes = set(df.index)
        file_gene_counts[name] = len(file_genes)
        common_genes = common_genes.intersection(file_genes)
        embeddings[name] = df

    if not common_genes:
        raise ValueError("No common genes found between AnnData and external embeddings.")

    common_genes_list = sorted(list(common_genes))
    n_dropped = len(adata_genes) - len(common_genes)

    r0_print("Gene intersection (static embeddings):")
    r0_print(f"  adata.var_names: {len(adata_genes)} genes")
    for name, n in file_gene_counts.items():
        r0_print(f"  {name}: {n} genes")
    r0_print(f"  intersection (static_embedding=True): {len(common_genes_list)} genes used for modeling")
    if n_dropped > 0:
        r0_print(f"  ({n_dropped} genes with static_embedding=False, excluded from modeling)")

    adata.var["static_embedding"] = np.asarray(adata.var_names.astype(str).isin(common_genes_list))

    aligned_embeddings = {}
    for name, df in embeddings.items():
        emb_subset = df.loc[common_genes_list]
        aligned_embeddings[name] = torch.tensor(emb_subset.values, dtype=torch.float32)

    return adata, aligned_embeddings
