"""
NDF Transformer Encoder for RL policies.

Processes NDF (Neural Descriptor Field) observations as a sequence of keypoint tokens:
each token = NDF feature at a keypoint + learned keypoint embedding. A transformer encoder
outputs updated latents per keypoint, which are flattened and fed to the rest of the policy.
"""

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class NDFMlpEncoder(nn.Module):
    """
    Debug drop-in replacement for NDFTransformerEncoder.

    - Input:  (B, num_keypoints * ndf_feature_dim)
    - Output: (B, output_dim)
    """

    def __init__(
        self,
        num_keypoints: int,
        ndf_feature_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.ndf_feature_dim = ndf_feature_dim
        self.output_dim = output_dim

        in_dim = num_keypoints * ndf_feature_dim
        hidden_dims = hidden_dims or []
        act = _get_activation(activation)

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, ndf_flat: torch.Tensor) -> torch.Tensor:
        return self.net(ndf_flat)


class NDFMinimalEncoder(nn.Module):
    """
    Transformer-like encoder with a single attention module (not a full transformer).
    
    Processes keypoints: project -> attention -> LayerNorm -> activation -> flatten.
    Uses a single attention layer to allow cross-keypoint interactions.
    
    - Input:  (B, num_keypoints * ndf_feature_dim)
    - Output: (B, num_keypoints * d_model)
    """

    def __init__(
        self,
        num_keypoints: int,
        ndf_feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        use_keypoint_embed: bool = False,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.ndf_feature_dim = ndf_feature_dim
        self.d_model = d_model
        self.use_keypoint_embed = use_keypoint_embed

        # Project each keypoint's NDF features to d_model
        self.ndf_proj = nn.Linear(ndf_feature_dim, d_model)

        # Optional keypoint embeddings (like transformer, but not needed for minimal)
        if self.use_keypoint_embed:
            self.keypoint_embed = nn.Parameter(torch.zeros(1, num_keypoints, d_model))
            nn.init.normal_(self.keypoint_embed, std=0.02)
        else:
            self.keypoint_embed = None

        # Single attention module (not a full transformer)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # LayerNorm per keypoint (normalizes across d_model dimension)
        # self.ln = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.act = _get_activation(activation)

        # Output dimension after flattening: num_keypoints * d_model
        self.output_dim = num_keypoints * d_model

    def forward(self, ndf_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ndf_flat: (batch_size, num_keypoints * ndf_feature_dim)

        Returns:
            (batch_size, num_keypoints * d_model) flattened latent for each keypoint
        """
        batch_size = ndf_flat.size(0)
        expected = self.num_keypoints * self.ndf_feature_dim
        if ndf_flat.dim() != 2 or ndf_flat.size(1) != expected:
            raise ValueError(
                f"NDFMinimalEncoder expected ndf_flat shape (B, {expected}) "
                f"but got {tuple(ndf_flat.shape)}; "
                f"num_keypoints={self.num_keypoints}, ndf_feature_dim={self.ndf_feature_dim}"
            )
        # Reshape to (B, num_keypoints, ndf_feature_dim)
        x = ndf_flat.reshape(batch_size, self.num_keypoints, self.ndf_feature_dim)
        # Project each keypoint independently to d_model
        x = self.ndf_proj(x)  # (B, num_keypoints, d_model)
        # Save input for residual connection (before keypoint embeddings)
        x_residual = x
        # Optional: add keypoint embeddings
        if self.keypoint_embed is not None:
            x = x + self.keypoint_embed
        # Apply attention module (allows cross-keypoint interactions)
        x, _ = self.attention(x, x, x)  # (B, num_keypoints, d_model)
        # LayerNorm per keypoint (normalizes across d_model dimension)
        # x = self.ln(x)  # (B, num_keypoints, d_model)
        # Activation
        x = self.act(x)  # (B, num_keypoints, d_model)
        # Add residual connection
        x = x + x_residual
        # Flatten back to (B, num_keypoints * d_model)
        return x.flatten(1)


class NDFTransformerEncoder(nn.Module):
    """
    Encodes NDF observations (multiple keypoints, each with a feature vector) using
    a transformer. Each token is ndf_feature + learned keypoint embedding.
    """

    def __init__(
        self,
        num_keypoints: int,
        ndf_feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.ndf_feature_dim = ndf_feature_dim
        self.d_model = d_model

        # Project NDF features to d_model (so we can add keypoint embeddings in same space)
        self.ndf_proj = nn.Linear(ndf_feature_dim, d_model)

        # Learned embeddings to indicate which keypoint each token is (position/keypoint id)
        self.keypoint_embed = nn.Parameter(torch.zeros(1, num_keypoints, d_model))
        nn.init.normal_(self.keypoint_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output dimension after flattening: num_keypoints * d_model
        self.output_dim = num_keypoints * d_model

    def forward(self, ndf_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ndf_flat: (batch_size, num_keypoints * ndf_feature_dim)

        Returns:
            (batch_size, num_keypoints * d_model) flattened latent for each keypoint
        """
        batch_size = ndf_flat.size(0)
        expected = self.num_keypoints * self.ndf_feature_dim
        if ndf_flat.dim() != 2 or ndf_flat.size(1) != expected:
            raise ValueError(
                f"NDFTransformerEncoder expected ndf_flat shape (B, {expected}) "
                f"but got {tuple(ndf_flat.shape)}; "
                f"num_keypoints={self.num_keypoints}, ndf_feature_dim={self.ndf_feature_dim}"
            )
        # Reshape to (B, num_keypoints, ndf_feature_dim)
        x = ndf_flat.reshape(batch_size, self.num_keypoints, self.ndf_feature_dim)
        # Project to d_model and add keypoint embeddings
        x = self.ndf_proj(x) + self.keypoint_embed  # (B, num_keypoints, d_model)
        # Transformer expects (B, seq_len, d_model); no need to transpose
        out = self.transformer(x)  # (B, num_keypoints, d_model)
        return out.flatten(1)  # (B, num_keypoints * d_model)
