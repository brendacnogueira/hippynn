"""Torch ViT-style transformer over HIP-HOP-NN invariant atom features."""

from __future__ import annotations

from typing import Any, Optional
import sys

import torch
import torch.nn as nn


from torch import functional as F
from hippynn.graphs.indextypes import IdxType
from hippynn.graphs.nodes.base import AutoKw, ExpandParents, SingleNode, find_unique_relative
from hippynn.graphs.nodes.tags import AtomIndexer, HAtomRegressor, Network, Positions
from hippynn.transformer.activation import MultiheadAttention

class IdentityLayer(nn.Module):
    """Identity layer, convenient for compatibility with older configs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AddPositionEmbs(nn.Module):
    """Learned absolute position embeddings for rank-3 token tensors."""

    def __init__(self, hidden_size: int, max_len: int = 512, init_std: float = 0.02):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, max_len, hidden_size))
        nn.init.normal_(self.pos_embedding, std=init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs with shape (batch, seq, hidden), got {tuple(inputs.shape)}")
        seq_len = inputs.shape[1]
        if seq_len > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.pos_embedding.shape[1]} "
                "for positional embeddings."
            )
        return inputs + self.pos_embedding[:, :seq_len].to(dtype=inputs.dtype, device=inputs.device)


class MlpBlock(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, hidden_size: int, mlp_dim: int, out_dim: Optional[int] = None, dropout_rate: float = 0.1):
        super().__init__()
        actual_out_dim = hidden_size if out_dim is None else out_dim
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, actual_out_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Encoder1DBlock(nn.Module):
    """Pre-norm transformer encoder layer."""

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        use_euclidean_rope: bool = False,
        use_distance_bias: bool = False,
        use_distance_mlp_bias: bool = False,
        rope_num_frequencies: int = 16,
        rope_min_frequency: float = 0.1,
        rope_max_frequency: float = 10.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            batch_first=True,
            use_euclidean_rope=use_euclidean_rope,
            use_distance_bias=use_distance_bias,
            use_distance_mlp_bias=use_distance_mlp_bias,
            rope_num_frequencies=rope_num_frequencies,
            rope_min_frequency=rope_min_frequency,
            rope_max_frequency=rope_max_frequency,
           
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MlpBlock(hidden_size=hidden_size, mlp_dim=mlp_dim, dropout_rate=dropout_rate)

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pairwise_distances: Optional[torch.Tensor] = None,
        pairwise_distance_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs with shape (batch, seq, hidden), got {tuple(inputs.shape)}")
        x_norm = self.norm1(inputs)
        attn_output, att_weights= self.attention(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            euclidean_rope_distances=pairwise_distances,
            euclidean_rope_mask=pairwise_distance_mask,
        )
        
        x = inputs + self.dropout(attn_output)
       
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    """Stack of 1D transformer encoder blocks."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        add_position_embedding: bool = False,
        max_len: int = 512,
        use_euclidean_rope: bool = True,
        use_distance_bias: bool = False,
        use_distance_mlp_bias: bool = False,
        rope_num_frequencies: int = 16,
        rope_min_frequency: float = 0.1,
        rope_max_frequency: float = 10.0,
        **_: Any,
    ):
        super().__init__()
        self.position_embedding = (
            AddPositionEmbs(hidden_size=hidden_size, max_len=max_len) if add_position_embedding else None
        )
        self.input_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            Encoder1DBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                use_euclidean_rope=use_euclidean_rope,
                use_distance_bias=use_distance_bias,
                use_distance_mlp_bias=use_distance_mlp_bias,
                rope_num_frequencies=rope_num_frequencies,
                rope_min_frequency=rope_min_frequency,
                rope_max_frequency=rope_max_frequency,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pairwise_distances: Optional[torch.Tensor] = None,
        pairwise_distance_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.position_embedding is not None:
            x = self.position_embedding(x)
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(
                x,
                key_padding_mask=key_padding_mask,
                pairwise_distances=pairwise_distances,
                pairwise_distance_mask=pairwise_distance_mask,
            )
        return self.norm(x)


def _atom_features_to_tokens(
    atom_features: torch.Tensor,
    system_index: torch.Tensor,
    n_systems: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad flat atom features into ``(systems, max_atoms, features)`` tokens."""

    if atom_features.ndim != 2:
        raise ValueError(f"Expected flat atom features with shape (atoms, features), got {tuple(atom_features.shape)}")

    n_systems_int = int(n_systems.item()) if torch.is_tensor(n_systems) else int(n_systems)
    counts = torch.bincount(system_index, minlength=n_systems_int)
    max_atoms = int(counts.max().item()) if counts.numel() else 0
    if max_atoms == 0:
        raise ValueError("Cannot build transformer tokens for a batch with no real atoms.")

    offsets = torch.cumsum(counts, dim=0) - counts
    atom_positions = torch.arange(atom_features.shape[0], device=atom_features.device) - offsets[system_index]

    tokens = atom_features.new_zeros((n_systems_int, max_atoms, atom_features.shape[-1]))
    valid_mask = torch.zeros((n_systems_int, max_atoms), dtype=torch.bool, device=atom_features.device)
    tokens[system_index, atom_positions] = atom_features
    valid_mask[system_index, atom_positions] = True
    return tokens, valid_mask


def _distances_to_token_pairs(
    positions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense pair distances for padded atom tokens."""

    pairwise_distances = torch.cdist(positions, positions)
    pairwise_distance_mask = valid_mask[:, :, None] & valid_mask[:, None, :]
    return pairwise_distances, pairwise_distance_mask


class VisionTransformer(nn.Module):
    """Small Torch ViT-style classifier over HIP-HOP-NN atom tokens."""

    def __init__(
        self,
        num_classes: int = 2,
        patches: Any = None,
        transformer: Optional[dict[str, Any]] = None,
        hidden_size: int = 20,
        resnet: Optional[Any] = None,
        representation_size: Optional[int] = None,
        classifier: str = "token",
        head_bias_init: float = 0.0,
        encoder: type[nn.Module] = Encoder,
        model_name: Optional[str] = None,
        feature_sizes: Optional[tuple[int, ...]] = None,
        feature_index: int = -1,
        max_len: int = 512,
        **_: Any,
    ):
        super().__init__()
        del patches, resnet, model_name

        if classifier not in {"token", "gap", "unpooled", "token_unpooled"}:
            raise ValueError(f"Invalid classifier={classifier!r}")

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.classifier = classifier
        self.feature_index = feature_index

        input_size = hidden_size
        if feature_sizes is not None:
            input_size = feature_sizes[feature_index]
        self.input_projection = nn.Identity() if input_size == hidden_size else nn.Linear(input_size, hidden_size)

        transformer = dict(transformer or {})
        transformer.setdefault("hidden_size", hidden_size)
        transformer.setdefault("mlp_dim", hidden_size * 4)
        transformer.setdefault("num_heads", 4)
        transformer.setdefault("num_layers", 2)
        transformer.setdefault("dropout_rate", 0.1)
        transformer.setdefault("attention_dropout_rate", 0.1)
        transformer.setdefault("add_position_embedding", False)
        transformer.setdefault("max_len", max_len)
        self.encoder = encoder(**transformer)

        if classifier in {"token", "token_unpooled"}:
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.cls = None

        if representation_size is None:
            self.pre_logits = IdentityLayer()
            head_in = hidden_size
        else:
            self.pre_logits = nn.Sequential(nn.Linear(hidden_size, representation_size), nn.Tanh())
            head_in = representation_size

        self.head = nn.Linear(head_in, num_classes) if num_classes else IdentityLayer()

        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, head_bias_init)

    def forward(
        self,
        hier_features: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        system_index: Optional[torch.Tensor] = None,
        atom_index: Optional[torch.Tensor] = None,
        n_systems: Optional[int | torch.Tensor] = None,
    ) -> torch.Tensor:
        if not _looks_like_hier_features(hier_features):
            supplied = (hier_features, system_index, n_systems)
            hier_features = next((value for value in supplied if _looks_like_hier_features(value)), hier_features)
            system_index = next((value for value in supplied if _looks_like_system_index(value)), system_index)
            n_systems = next((value for value in supplied if _looks_like_n_systems(value)), n_systems)
       
        if isinstance(hier_features, (list, tuple)):
            atom_features = hier_features[self.feature_index]
        else:
            atom_features = hier_features
        if system_index is None or n_systems is None:
            if atom_features.ndim != 3:
                raise ValueError("Direct tensor input must have shape (batch, seq, features).")
            x = atom_features
            
            valid_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            token_positions = positions
        else:
            x, valid_mask = _atom_features_to_tokens(atom_features, system_index, n_systems)
            token_positions = None
            if positions is not None:
                if positions.ndim == 3:
                    if atom_index is None:
                        raise ValueError("atom_index is required to align batched positions with flat atom features.")
                    positions = positions[system_index, atom_index]
                token_positions, _ = _atom_features_to_tokens(positions, system_index, n_systems)
        
        x = self.input_projection(x)
        key_padding_mask = ~valid_mask
        pairwise_distances = None
        pairwise_distance_mask = None
        if token_positions is not None:
            if token_positions.ndim != 3:
                raise ValueError("Position input must have shape (batch, seq, 3) after tokenization.")
            pairwise_distances, pairwise_distance_mask = _distances_to_token_pairs(token_positions, valid_mask)

        if self.cls is not None:
            cls = self.cls.expand(x.shape[0], -1, -1).to(dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
            cls_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
            if pairwise_distances is not None:
                padded_distances = pairwise_distances.new_zeros((x.shape[0], x.shape[1], x.shape[1]))
                padded_distances[:, 1:, 1:] = pairwise_distances
                pairwise_distances = padded_distances
                padded_pair_mask = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device)
                padded_pair_mask[:, 1:, 1:] = pairwise_distance_mask
                pairwise_distance_mask = padded_pair_mask

        x = self.encoder(
            x,

            key_padding_mask=key_padding_mask,
            pairwise_distances=pairwise_distances,
            pairwise_distance_mask=pairwise_distance_mask,
        )

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "gap":
            weights = (~key_padding_mask).to(dtype=x.dtype).unsqueeze(-1)
            x = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        elif self.classifier in {"unpooled", "token_unpooled"}:
            pass


        
        pre_logits = self.pre_logits(x)
        logits = self.head(pre_logits)
        probs = torch.softmax(logits, dim=-1)

        
        self._debug_output_count = 0

        if (self._debug_output_count < 10) and (False):
            with torch.no_grad():
                print("\n[final transformer output debug]")
                print(f"encoder output x shape: {tuple(x.shape)}")
                print(f"pre_logits shape:       {tuple(pre_logits.shape)}")
                print(f"logits shape:           {tuple(logits.shape)}")

                for mol in range(min(x.shape[0], 2)):
                    print(f"\nmolecule {mol}")
                    print("encoder pooled/final x:")
                    print(x[mol].detach().cpu())

                    print("pre_logits:")
                    print(pre_logits[mol].detach().cpu())

                    print("logits:")
                    print(logits[mol].detach().cpu())

                    print("probabilities:")
                    print(probs[mol].detach().cpu())

                if x.shape[0] >= 2:
                    print("\ndifferences molecule 0 - molecule 1")
                    print("x abs max diff:", (x[0] - x[1]).detach().abs().max().cpu().item())
                    print("x L2 diff:", torch.linalg.vector_norm((x[0] - x[1]).detach()).cpu().item())
                    print("pre_logits abs max diff:", (pre_logits[0] - pre_logits[1]).detach().abs().max().cpu().item())
                    print("pre_logits L2 diff:", torch.linalg.vector_norm((pre_logits[0] - pre_logits[1]).detach()).cpu().item())
                    print("logits diff:", (logits[0] - logits[1]).detach().cpu())
                
                

            self._debug_output_count += 1

        return logits
        #return self.head(self.pre_logits(x))


def _looks_like_hier_features(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return True
    return torch.is_tensor(value) and value.ndim >= 2 and torch.is_floating_point(value)


def _looks_like_system_index(value: Any) -> bool:
    return torch.is_tensor(value) and value.ndim == 1 and value.dtype in (torch.int32, torch.int64)


def _looks_like_atom_index(value: Any) -> bool:
    return _looks_like_system_index(value)


def _looks_like_n_systems(value: Any) -> bool:
    return isinstance(value, int) or (torch.is_tensor(value) and value.ndim == 0)


class TransformerNode(HAtomRegressor, ExpandParents, AutoKw, SingleNode):
    """Hippynn graph node that classifies systems from HIP-HOP-NN features."""

    input_names = "hier_features", "positions", "system_index", "atom_index", "n_systems"
    index_state = IdxType.Systems
    auto_module_class = VisionTransformer
    auto_module_kwargs = "num_classes", "transformer", "hidden_size", "classifier", "feature_index"

    @parent_expander.match(Network)
    def expansion0(self, net, **kwargs):
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = net.torch_module.feature_sizes
        pidxer = find_unique_relative(net, AtomIndexer)
        positions = find_unique_relative(net, Positions)
        return net, positions, pidxer.system_index, pidxer.atom_index, pidxer.n_systems

    parent_expander.assertlen(5)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(None, None, None, None, None)

    def __init__(self, name, parents, module="auto", module_kwargs=None, **kwargs):
        self.module_kwargs = module_kwargs or {}
        super().__init__(name, parents, module=module, **kwargs)


KChainsTransformerNode = TransformerNode
