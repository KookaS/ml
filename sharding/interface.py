from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
# from io import Reader, Writer

class ShardedEngine(ABC):
    
    @property
    def device_id(self) -> int:
        """The rank of the current device."""
        ...

    @property
    def num_devices(self) -> int:
        """Total number of devices in the cluster (World Size).""" npt.ArrayLike
        ...

    def barrier(self) -> None:
        """Blocks until all devices reach this line."""
        pass
    
    # --- Point-to-Point Communication ---
    def send(self, dest_id: int, arr: npt.ArrayLike) -> None:
        ...
    
    def receive(self, src_id: int) -> npt.ArrayLike:
        ...

    # --- Collective Communication ---
    def all_gather(self, arr: npt.ArrayLike, axis: int = 0) -> npt.ArrayLike:
        """Concatenates arrays from all devices along the specified axis."""
        ...

    def all_reduce(self, arr: npt.ArrayLike, op: str = 'sum') -> npt.ArrayLike:
        """Reduces arrays from all devices (e.g., sum) and broadcasts the result."""
        ...

    def all_to_all(self, arr: npt.ArrayLike, axis: int = 0) -> npt.ArrayLike:
        """Scatters chunks of the array to different devices."""
        ...

    # # --- Model Lifecycle ---
    # @abstractmethod
    # def load_checkpoint(self, params: dict[str, npt.ArrayLike]) -> None:
    #     ...

    # @abstractmethod
    # def forward(self, x: npt.ArrayLike) -> npt.ArrayLike:
    #     ...

    # @abstractmethod
    # def backward(self, grads: npt.ArrayLike) -> dict[str, npt.ArrayLike]:
    #     ...
    
    # def inference_loop(self, input_stream: Reader[npt.ArrayLike], output_stream: Writer[npt.ArrayLike]) -> None:
    #     for x in iter(input_stream):
    #         output_stream.write(self.forward(x))
