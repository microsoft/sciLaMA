import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
import pandas as pd
from .utils import check_normalized_scaled, load_and_match_feature_embeddings, fit_covariate_encoding_state, encode_covariates, CovariateEncodingState, _categorical_covariate_columns
from .config import DataConfig, TrainingConfig


class RNADataset(Dataset):
    def __init__(self, X: torch.Tensor, C: torch.Tensor):
        self.X = X
        self.C = C

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx]


class SciLaMADataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig, training_config: TrainingConfig):
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.adata = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.feature_input_embeddings = None
        self.n_features = 0
        self.n_covariates = 0
        self._covariate_encoding_state: None | CovariateEncodingState = None

    def get_covariate_encoding_state(self) -> None | CovariateEncodingState:
        """Encoding state (one-hot mapping, continuous mean/std) to save in checkpoint for inference."""
        return self._covariate_encoding_state

    def set_covariate_encoding_state(self, state: None | CovariateEncodingState) -> None:
        """Set encoding state when loading from checkpoint for predict (same encoding as training)."""
        self._covariate_encoding_state = state

    def setup(self, stage: str | None = None):
        from .utils import r0_print, r0_rich
        # load data
        r0_print(f"Loading data from {self.config.path}...")
        try:
            self.adata = sc.read_h5ad(self.config.path)
        except Exception as e:
            raise ValueError(f"Failed to load data from {self.config.path}: {e}")

        # check scaling
        if self.config.check_scaling:
            if not check_normalized_scaled(self.adata):
                r0_print("Warning: Data might not be normalized/scaled correctly for MSE loss.")

        # handle external feature embeddings
        if self.config.external_feature_embeddings:
            r0_print("Loading external feature embeddings...")
            self.adata, self.feature_input_embeddings = load_and_match_feature_embeddings(
                self.adata, self.config.external_feature_embeddings
            )
        else:
            r0_print("No external feature embeddings provided. Using all features.")
            self.adata.var["static_embedding"] = True
            self.feature_input_embeddings = {}

        # use only genes with static_embedding=True for modeling (no adata subsetting)
        static_mask = self.adata.var["static_embedding"].values
        self.n_features = int(static_mask.sum())
        r0_print(f"Modeling {self.n_features} features (static_embedding=True).")

        # split indices (need train first to fit covariate encoding from train only)
        if self.config.split_column not in self.adata.obs:
            raise ValueError(f"Split column '{self.config.split_column}' not found in obs.")
        split_col = self.adata.obs[self.config.split_column]
        train_idx = split_col == self.config.train_split_key
        val_idx = split_col == self.config.val_split_key
        test_idx = (split_col == self.config.test_split_key) if self.config.test_split_key is not None else pd.Series(False, index=split_col.index)
        if train_idx.sum() == 0:
            raise ValueError(f"No '{self.config.train_split_key}' samples in obs['{self.config.split_column}'].")
        if val_idx.sum() == 0:
            raise ValueError(f"No '{self.config.val_split_key}' samples in obs['{self.config.split_column}']. Val is required for early stopping.")

        # covariates: discrete (one-hot) + continuous (z-score). fit state on train only; save in checkpoint for inference.
        cat_cols = _categorical_covariate_columns(self.config)
        has_cont = bool(self.config.continuous_covariate_keys)
        if cat_cols or has_cont:
            if self._covariate_encoding_state is None:
                train_obs = self.adata.obs.loc[train_idx]
                self._covariate_encoding_state = fit_covariate_encoding_state(train_obs, self.config)
                if cat_cols:
                    r0_print(f"Encoding discrete covariates (one-hot): {cat_cols}")
                if has_cont:
                    r0_print(f"Encoding continuous covariates (z-score): {self.config.continuous_covariate_keys}")
            covariates = encode_covariates(self.adata.obs, self._covariate_encoding_state, self.config)
            covariates = torch.FloatTensor(covariates)
            self.n_covariates = covariates.shape[1]
        else:
            self.n_covariates = 0
            covariates = torch.zeros((self.adata.n_obs, 0))

        # prepare X and datasets (only static_embedding genes for modeling)
        X = self.adata[:, static_mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = torch.FloatTensor(X)
        train_mask = train_idx.to_numpy()
        val_mask = val_idx.to_numpy()
        test_mask = test_idx.to_numpy()
        self.train_dataset = RNADataset(X[train_mask], covariates[train_mask])
        self.val_dataset = RNADataset(X[val_mask], covariates[val_mask])
        if test_mask.sum() > 0:
            self.test_dataset = RNADataset(X[test_mask], covariates[test_mask])

        r0_rich(
            f"[green]Data setup complete:[/green] Train={len(self.train_dataset) if self.train_dataset else 0}, "
            f"Val={len(self.val_dataset) if self.val_dataset else 0}, "
            f"Test={len(self.test_dataset) if self.test_dataset else 0}"
        )
        r0_rich(f"Features: [bold]{self.n_features}[/bold], Covariates: [bold]{self.n_covariates}[/bold]")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        return None

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        return None

    def predict_dataloader(self):
        static_mask = self.adata.var["static_embedding"].values
        X = self.adata[:, static_mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = torch.FloatTensor(X)
        if self.n_covariates > 0 and self._covariate_encoding_state is not None:
            covariates = torch.FloatTensor(encode_covariates(self.adata.obs, self._covariate_encoding_state, self.config))
        else:
            covariates = torch.zeros((self.adata.n_obs, 0))
        full_dataset = RNADataset(X, covariates)
        return DataLoader(
            full_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
