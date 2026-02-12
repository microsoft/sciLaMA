from typing import List, Literal
from pydantic import BaseModel

class DataConfig(BaseModel):
    path: str
    # Discrete (one-hot): list of obs columns. Encoding state (category order) saved in checkpoint.
    categorical_covariate_keys: List[str] | None = None
    # Continuous (z-score scaled): mean/std from train data are saved in checkpoint.
    continuous_covariate_keys: List[str] | None = None
    split_column: str = "split"  # obs column containing split labels
    train_split_key: str = "train"
    val_split_key: str = "val"  # used for early stopping
    test_split_key: str | None = "test"  # optional; None means no test split
    external_feature_embeddings: dict[str, str] | None = None
    check_scaling: bool = True

class ModelConfig(BaseModel):
    hidden_dims: List[int] = [900, 400]
    latent_dim: int = 50
    dropout_rate: float = 0.1
    batchnorm: bool = False
    layernorm: bool = True
    activation: str = "LeakyReLU"
    fusion_method: Literal["average", "MoE", "PoE"] = "average"
    var_eps: float = 1e-4

class TrainingConfig(BaseModel):
    seed: int = 42
    mode: Literal["direct", "stepwise", "beta_vae"] = "direct"
    max_epochs: int = 500
    batch_size: int = 128
    # Multi-GPU: devices=2 or "auto" for all GPUs; strategy="ddp" for distributed
    devices: int | str = 1
    strategy: Literal["auto", "ddp", "ddp_find_unused_parameters_false"] = "auto"
    learning_rate: float = 0.001
    patience: int = 25
    weight_decay: float = 0.0
    beta_start: float = 0.0
    beta_end: float = 1.0
    epochs_before_beta_warmup: int = 25  # epochs with KL weight = beta_start (no KL)
    beta_warmup_rate: float = 0.05   # per-epoch increase after warmup (until beta_end)
    gamma: float = 0.05
    val_check_interval: float = 1  # fraction of train dataloader (e.g. 0.5 = validate 2x per epoch; 2 = validate every 2 epochs)

class OutputConfig(BaseModel):
    save_dir: str = "./results"
    save_key: str = "X_sciLaMA"  # sample embeddings in adata.obsm[save_key] and parquet (mode-specific names)
    save_top_k: int = 1 # keep top k checkpoints (sorted by val loss)

class SciLaMAConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig
