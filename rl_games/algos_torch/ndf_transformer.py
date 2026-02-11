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
    Can be conditioned on robot/object states by adding them as additional tokens.
    
    - Input:  (B, num_keypoints * ndf_feature_dim)
    - Optional: robot_obj_states (B, state_dim) - robot and object states (e.g., indices 0:36)
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
        robot_obj_state_dim: int = 36,
        attention_ndf_only: bool = True,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.ndf_feature_dim = ndf_feature_dim
        self.d_model = d_model
        self.use_keypoint_embed = use_keypoint_embed
        self.robot_obj_state_dim = robot_obj_state_dim
        # If True, attention only sees NDF keypoint tokens
        # and ignores robot/object state tokens even if provided.
        self.attention_ndf_only = attention_ndf_only

        # Project each keypoint's NDF features to d_model
        self.ndf_proj = nn.Linear(ndf_feature_dim, d_model)

        # Project robot/object states to d_model (if provided)
        if robot_obj_state_dim > 0 and not attention_ndf_only:
            self.robot_obj_proj = nn.Linear(robot_obj_state_dim, d_model)
        else:
            self.robot_obj_proj = None

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

    def forward(self, ndf_flat: torch.Tensor, robot_obj_states: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            ndf_flat: (batch_size, num_keypoints * ndf_feature_dim)
            robot_obj_states: Optional (batch_size, robot_obj_state_dim) - robot and object states
                             (e.g., indices 0:36 from rest_obs). If None, attention is not conditioned.

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
        
        # Condition attention on robot/object states if provided and not disabled
        if (
            not self.attention_ndf_only
            and robot_obj_states is not None
            and self.robot_obj_proj is not None
        ):
            # Project robot/object states to d_model
            robot_obj_proj = self.robot_obj_proj(robot_obj_states)  # (B, d_model)
            # Expand to (B, 1, d_model) to add as an additional token
            robot_obj_proj = robot_obj_proj.unsqueeze(1)  # (B, 1, d_model)
            
            # Concatenate robot/object states with keypoints for attention
            # Now x has shape (B, num_keypoints + 1, d_model)
            x_with_states = torch.cat([x, robot_obj_proj], dim=1)  # (B, num_keypoints + 1, d_model)
            
            # Apply attention: keypoints can attend to robot/object states and vice versa
            x_attended, _ = self.attention(x_with_states, x_with_states, x_with_states)  # (B, num_keypoints + 1, d_model)
            
            # Extract only the keypoint tokens (first num_keypoints tokens)
            x = x_attended[:, :self.num_keypoints, :]  # (B, num_keypoints, d_model)
        else:
            # Apply attention module (allows cross-keypoint interactions only)
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
