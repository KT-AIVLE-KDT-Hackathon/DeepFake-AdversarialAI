# net_info.py
from dataclasses import dataclass, field
from typing import Callable
from tensorflow.keras.layers import Layer
import typing as T

@dataclass
class NetInfo:
    model_id: int = 0
    model_name: str = ""
    net: Callable | None = None
    init_kwargs: dict[str, T.Any] = field(default_factory=dict)
    needs_init: bool = True
    outputs: list[Layer] = field(default_factory=list)
