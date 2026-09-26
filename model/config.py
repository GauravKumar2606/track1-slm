from dataclasses import dataclass

import yaml


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float
    batch_size: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    checkpoint_every: int

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # PyYAML doesn't recognize exponent notation without a decimal point
        # (e.g. "3e-4") as a float and leaves it as a string — coerce
        # explicitly rather than relying on the YAML file's exact formatting.
        data["dropout"] = float(data["dropout"])
        data["learning_rate"] = float(data["learning_rate"])
        return cls(**data)
