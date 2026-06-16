"""Public entry points for the local Torch transformer models."""

from hippynn.transformer import models_vit

AddPositionEmbs = models_vit.AddPositionEmbs
MlpBlock = models_vit.MlpBlock
Encoder1DBlock = models_vit.Encoder1DBlock
Encoder = models_vit.Encoder
VisionTransformer = models_vit.VisionTransformer
KChainsTransformerNode = models_vit.KChainsTransformerNode


def get_model(name=None, **kw):
    """Return a Torch ``VisionTransformer``.

    ``name`` is accepted for compatibility with the older ViT-JAX-style API.
    """

    del name
    return VisionTransformer(**kw)


def config():
    """Small default config-like dictionary for smoke tests."""

    return {
        "batch": 64,
        "batch_eval": 8,
        "total_steps": 1,
        "model": {
            "num_classes": 2,
            "hidden_size": 20,
            "transformer": {
                "mlp_dim": 80,
                "num_heads": 4,
                "num_layers": 2,
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.0,
                "add_position_embedding": False,
            },
            "classifier": "token",
        },
    }
