from typing import Callable, Dict, Type

import torch.nn as nn


class ModelRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[nn.Module]] = {}

    def register(self, name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
            self._registry[name] = model_class
            return model_class

        return decorator

    def get(self, name: str) -> Type[nn.Module]:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry)) or "none"
            raise ValueError(f"Unknown model '{name}'. Available models: {available}")
        return self._registry[name]


MODEL_REGISTRY = ModelRegistry()
