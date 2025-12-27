"""
JAX Sharded Engine - A wrapper providing MPI-style distributed primitives over JAX.

This module provides an object-oriented interface to JAX's distributed computing
primitives, supporting multi-dimensional mesh topologies for advanced parallelism
strategies (data parallel, tensor parallel, pipeline parallel, and combinations).

Compatible with JAX's modern mesh APIs:
    - jax.make_mesh() for mesh creation
    - jax.set_mesh() for global mesh context
    - jax.NamedSharding and jax.P for array placement

Example:
    >>> engine = JaxShardedEngine(
    ...     axis_shapes=(2, 4),
    ...     axis_names=('dp', 'tp'),
    ... )
    >>> engine.set_global_mesh()
    >>> 
    >>> # Create sharded arrays using engine's mesh
    >>> x = jnp.zeros((8, 1024), device=engine.sharding(P('dp', 'tp')))
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
import numpy as np
import numpy.typing as npt


ReduceOp = Literal['sum', 'mean', 'max', 'min', 'prod']
AxisName = str | tuple[str, ...]


class JaxShardedEngine:
    """
    A distributed computing engine supporting multi-dimensional device meshes.
    
    This class wraps JAX's functional distributed primitives into an object-oriented
    interface, supporting complex parallelism strategies through multi-axis meshes.
    Common configurations include:
    
    - 1D mesh ('dp',): Pure data parallelism
    - 2D mesh ('dp', 'tp'): Data + tensor parallelism
    - 3D mesh ('dp', 'tp', 'pp'): Data + tensor + pipeline parallelism
    
    Attributes:
        axis_names: Tuple of axis name strings for the mesh dimensions.
        axis_shapes: Tuple of integers defining device count per axis.
        mesh: JAX Mesh object defining the device topology.
        
    Example:
        >>> # 2D mesh: 2 data-parallel × 4 tensor-parallel
        >>> engine = JaxShardedEngine(
        ...     axis_shapes=(2, 4),
        ...     axis_names=('dp', 'tp'),
        ... )
        >>> engine.set_global_mesh()
        >>> 
        >>> # Create sharded arrays
        >>> x = jnp.zeros((8, 1024), device=engine.sharding(P('dp', 'tp')))
        >>> w = jnp.zeros((1024, 2048), device=engine.sharding(P(None, 'tp')))
        >>> 
        >>> # Collective across specific axis
        >>> @engine.sharded(in_specs=P('dp', 'tp'), out_specs=P('dp', 'tp'))
        ... def sync_across_dp(x):
        ...     return engine.all_reduce(x, axis='dp', op='mean')
    
    Note:
        For multi-host setups, ensure environment variables are set:
        - COORDINATOR_ADDRESS: IP:port of coordinator
        - NUM_PROCESSES: Total host count
        - PROCESS_ID: This host's rank
    """
    
    def __init__(
        self,
        axis_shapes: tuple[int, ...] | None = None,
        axis_names: tuple[str, ...] | None = None,
        devices: Sequence[jax.Device] | None = None,
        initialize_distributed: bool = False,
    ) -> None:
        """
        Initialize the sharded engine with a multi-dimensional device mesh.
        
        Creates a mesh topology mapping logical axis names to physical devices.
        The product of axis_shapes must equal the number of available devices.
        
        Args:
            axis_shapes: Shape of the mesh as a tuple of integers. Each element
                defines how many devices along that axis. The product must
                equal the total device count.
                Examples:
                - (8,) for 8 devices in 1D
                - (2, 4) for 2×4 = 8 devices in 2D
                - (2, 2, 2) for 2×2×2 = 8 devices in 3D
                Default: (num_devices,) - all devices in 1D
                
            axis_names: Names for each mesh axis, must match len(axis_shapes).
                Common conventions:
                - 'dp' / 'data': Data parallelism axis
                - 'tp' / 'tensor' / 'mp': Tensor/model parallelism axis
                - 'pp' / 'pipeline': Pipeline parallelism axis
                - 'fsdp': Fully-sharded data parallel axis
                - 'X', 'Y', 'Z': Generic axis names
                Default: ('devices',) for 1D mesh
                
            devices: Optional specific devices to use. If None, uses all
                available devices from jax.devices().
                
            initialize_distributed: Whether to call jax.distributed.initialize()
                for multi-host setups. Set True when running across multiple
                machines. Default: False
        
        Raises:
            ValueError: If axis_shapes product doesn't match device count,
                or if axis_names length doesn't match axis_shapes length.
        
        Example:
            >>> # Single axis (data parallel only)
            >>> engine = JaxShardedEngine(
            ...     axis_shapes=(8,),
            ...     axis_names=('dp',),
            ... )
            
            >>> # Two axes (data + tensor parallel)
            >>> engine = JaxShardedEngine(
            ...     axis_shapes=(2, 4),
            ...     axis_names=('dp', 'tp'),
            ... )
            
            >>> # Three axes (data + tensor + pipeline)
            >>> engine = JaxShardedEngine(
            ...     axis_shapes=(2, 2, 2),
            ...     axis_names=('dp', 'tp', 'pp'),
            ... )
            
            >>> # Compatible with jax.make_mesh style
            >>> engine = JaxShardedEngine(
            ...     axis_shapes=(1, 1),
            ...     axis_names=('X', 'Y'),
            ... )
        """
        if initialize_distributed:
            jax.distributed.initialize()
        
        self._devices = list(devices) if devices is not None else jax.devices()
        num_devices = len(self._devices)
        
        # Default to 1D mesh if not specified
        if axis_shapes is None:
            axis_shapes = (num_devices,)
        if axis_names is None:
            axis_names = ('devices',) if len(axis_shapes) == 1 else tuple(
                f'axis_{i}' for i in range(len(axis_shapes))
            )
        
        # Validation
        if len(axis_shapes) != len(axis_names):
            raise ValueError(
                f"axis_shapes length ({len(axis_shapes)}) must match "
                f"axis_names length ({len(axis_names)})"
            )
        
        shape_product = np.prod(axis_shapes)
        if shape_product != num_devices:
            raise ValueError(
                f"Product of axis_shapes {axis_shapes} = {shape_product} "
                f"must equal number of devices ({num_devices})"
            )
        
        self._axis_shapes = axis_shapes
        self._axis_names = axis_names
        
        # Create mesh using jax.make_mesh (modern API)
        self._mesh = jax.make_mesh(
            axis_shapes=axis_shapes,
            axis_names=axis_names,
            devices=self._devices,
        )
        
        # Reduction operations mapping
        self._reduce_ops: dict[str, Callable] = {
            'sum': lax.psum,
            'mean': lax.pmean,
            'max': lax.pmax,
            'min': lax.pmin,
            # 'prod': lax.pprod,
        }
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def axis_names(self) -> tuple[str, ...]:
        """
        Names of all mesh axes.
        
        Returns:
            tuple[str, ...]: Axis names in order (e.g., ('dp', 'tp')).
        
        Example:
            >>> engine = JaxShardedEngine(axis_shapes=(2, 4), axis_names=('dp', 'tp'))
            >>> engine.axis_names
            ('dp', 'tp')
        """
        return self._axis_names
    
    @property
    def axis_shapes(self) -> tuple[int, ...]:
        """
        Size of each mesh axis.
        
        Returns:
            tuple[int, ...]: Number of devices along each axis.
        
        Example:
            >>> engine = JaxShardedEngine(axis_shapes=(2, 4), axis_names=('dp', 'tp'))
            >>> engine.axis_shapes
            (2, 4)
        """
        return self._axis_shapes
    
    @property
    def mesh(self) -> jax.sharding.Mesh:
        """
        The JAX Mesh object defining device topology.
        
        Use this for advanced sharding configurations, with shard_map,
        or when you need direct mesh access.
        
        Returns:
            Mesh: Multi-dimensional device mesh.
        
        Example:
            >>> sharding = jax.NamedSharding(engine.mesh, P('dp', None))
        """
        return self._mesh
    
    @property
    def device_id(self) -> int:
        """
        The rank/index of the current process (host) in the cluster.
        
        Returns:
            int: Zero-indexed process rank (0 to num_processes - 1).
        
        Note:
            This is the *process* index, not device index. Use
            `local_device_id(axis)` inside sharded functions for
            per-axis device indices.
        """
        return jax.process_index()
    
    @property
    def num_devices(self) -> int:
        """
        Total number of devices across all hosts.
        
        Returns:
            int: Total accelerator count (product of axis_shapes).
        """
        return len(self._devices)
    
    @property
    def num_processes(self) -> int:
        """
        Total number of processes (hosts) in the distributed setup.
        
        Returns:
            int: Number of JAX processes/hosts.
        """
        return jax.process_count()
    
    @property
    def num_local_devices(self) -> int:
        """
        Number of devices controlled by this process/host.
        
        Returns:
            int: Local accelerator count.
        """
        return jax.local_device_count()
    
    def axis_size(self, axis: str) -> int:
        """
        Get the size of a specific mesh axis.
        
        Args:
            axis: Name of the axis to query.
        
        Returns:
            int: Number of devices along that axis.
        
        Raises:
            ValueError: If axis name not in mesh.
        
        Example:
            >>> engine = JaxShardedEngine(axis_shapes=(2, 4), axis_names=('dp', 'tp'))
            >>> engine.axis_size('dp')
            2
            >>> engine.axis_size('tp')
            4
        """
        if axis not in self._axis_names:
            raise ValueError(
                f"Unknown axis '{axis}'. Available: {self._axis_names}"
            )
        idx = self._axis_names.index(axis)
        return self._axis_shapes[idx]
    
    # =========================================================================
    # Global Mesh Management
    # =========================================================================
    
    def set_global_mesh(self) -> None:
        """
        Deprecated: Use 'with engine:' context manager instead.

        This method is a no-op for backwards compatibility.

        Example:
            >>> engine = JaxShardedEngine(
            ...     axis_shapes=(2, 4),
            ...     axis_names=('X', 'Y'),
            ... )
            >>> with engine:
            ...     x = jnp.zeros((8, 1024), device=engine.sharding(P('X', 'Y')))
            ...     w = jnp.zeros((1024, 2048), device=engine.sharding(P(None, 'Y')))
        """
        pass  # Use context manager instead

    def clear_global_mesh(self) -> None:
        """
        Deprecated: Use 'with engine:' context manager instead.

        This method is a no-op for backwards compatibility.
        """
        pass  # Use context manager instead
    
    # =========================================================================
    # Synchronization
    # =========================================================================
    
    def barrier(self, name: str = 'barrier') -> None:
        """
        Block until all processes reach this synchronization point.
        
        Creates a global barrier across all hosts. Essential for coordinating
        operations like checkpointing where all processes must synchronize.
        
        Args:
            name: Unique identifier for this barrier. Different barriers
                should have different names to prevent deadlocks.
        
        Warning:
            All processes MUST call the same barriers in the same order.
        
        Example:
            >>> engine.barrier('pre_checkpoint')
            >>> if engine.device_id == 0:
            ...     save_checkpoint(params)
            >>> engine.barrier('post_checkpoint')
        """
        multihost_utils.sync_global_devices(name)
    
    # =========================================================================
    # Collective Communication Primitives
    # =========================================================================
    
    def all_gather(
        self,
        arr: jax.Array,
        axis: AxisName,
        *,
        gather_axis: int = 0,
        tiled: bool = False,
    ) -> jax.Array:
        """
        Gather arrays from all devices along specified mesh axis/axes.
        
        Concatenates array shards from devices along the given mesh axis.
        All devices receive the complete gathered result.
        
        Args:
            arr: Local array shard on this device.
            axis: Mesh axis name(s) to gather across. Can be a single
                string or tuple of strings for multi-axis gather.
            gather_axis: Array axis along which to concatenate gathered
                data. Default: 0
            tiled: If True, output shape along gather_axis equals input
                shape. If False, output expands by factor of axis size.
        
        Returns:
            jax.Array: Gathered array, replicated across the specified
                mesh axis/axes.
        
        Diagram (2×4 mesh, gather across 'dp'):
            Before (dp=0):  [A0, A1, A2, A3]    After: [A0, A1, A2, A3,
            Before (dp=1):  [B0, B1, B2, B3]            B0, B1, B2, B3]
            (4 tp-devices each)                 (all 8 devices see this)
        
        Example:
            >>> @engine.sharded(in_specs=P('dp', 'tp'), out_specs=P(None, 'tp'))
            ... def gather_across_dp(x):
            ...     return engine.all_gather(x, axis='dp', gather_axis=0)
            
            >>> # Multi-axis gather
            >>> @engine.sharded(in_specs=P('dp', 'tp'), out_specs=P())
            ... def gather_all(x):
            ...     return engine.all_gather(x, axis=('dp', 'tp'))
        
        Note:
            Must be called within a parallel context (pmap/shard_map).
        """
        return lax.all_gather(arr, axis, axis=gather_axis, tiled=tiled)
    
    def all_reduce(
        self,
        arr: jax.Array,
        axis: AxisName,
        op: ReduceOp = 'sum',
    ) -> jax.Array:
        """
        Reduce arrays across devices along specified mesh axis/axes.
        
        Applies a reduction operation element-wise across the given mesh
        axis. All devices receive the identical reduced result.
        
        Args:
            arr: Local array on this device.
            axis: Mesh axis name(s) to reduce across. Can be a single
                string or tuple for multi-axis reduction.
            op: Reduction operation: 'sum', 'mean', 'max', 'min', 'prod'.
                Default: 'sum'
        
        Returns:
            jax.Array: Reduced array with same shape as input.
        
        Raises:
            KeyError: If op is not recognized.
        
        Diagram (2×4 mesh, reduce across 'dp' with sum):
            dp=0, tp=0: [1, 2]  ┐                   
            dp=1, tp=0: [3, 4]  ┴─→ tp=0 gets: [4, 6]
            
            dp=0, tp=1: [5, 6]  ┐                   
            dp=1, tp=1: [7, 8]  ┴─→ tp=1 gets: [12, 14]
        
        Example:
            >>> # Sync gradients across data-parallel axis only
            >>> @engine.sharded(in_specs=P('dp', 'tp'), out_specs=P('dp', 'tp'))
            ... def sync_grads_dp(grads):
            ...     return engine.all_reduce(grads, axis='dp', op='mean')
            
            >>> # Reduce across both axes
            >>> @engine.sharded(in_specs=P('dp', 'tp'), out_specs=P())
            ... def global_sum(x):
            ...     return engine.all_reduce(x, axis=('dp', 'tp'), op='sum')
        
        Use Cases:
            - Gradient averaging across data-parallel replicas
            - Global statistics computation
            - Loss aggregation
        
        Note:
            Must be called within a parallel context (pmap/shard_map).
        """
        reduce_fn = self._reduce_ops.get(op)
        if reduce_fn is None:
            raise KeyError(
                f"Unknown reduction op '{op}'. "
                f"Available: {list(self._reduce_ops.keys())}"
            )
        return reduce_fn(arr, axis)
    
    def all_to_all(
        self,
        arr: jax.Array,
        axis: str,
        split_axis: int = 0,
        concat_axis: int = 0,
    ) -> jax.Array:
        """
        Redistribute array chunks across devices along a mesh axis.
        
        Splits each device's array along split_axis into N chunks (N =
        axis size), exchanges chunks so chunk i goes to device i along
        that mesh axis, then concatenates along concat_axis.
        
        Args:
            arr: Local array. split_axis size must be divisible by the
                mesh axis size.
            axis: Single mesh axis name to perform all-to-all across.
            split_axis: Array axis to split into chunks. Default: 0
            concat_axis: Array axis to concatenate received chunks. Default: 0
        
        Returns:
            jax.Array: Redistributed array.
        
        Diagram (4 devices along 'tp' axis):
            Before:                     After:
            tp=0: [A0, A1, A2, A3]      tp=0: [A0, B0, C0, D0]
            tp=1: [B0, B1, B2, B3]  →   tp=1: [A1, B1, C1, D1]
            tp=2: [C0, C1, C2, C3]      tp=2: [A2, B2, C2, D2]
            tp=3: [D0, D1, D2, D3]      tp=3: [A3, B3, C3, D3]
        
        Example:
            >>> # Switch from batch-sharded to head-sharded
            >>> @engine.sharded(
            ...     in_specs=P('dp', None, None),
            ...     out_specs=P(None, 'dp', None)
            ... )
            ... def reshard_batch_to_heads(x):
            ...     # x: (batch/dp, heads, dim) → (batch, heads/dp, dim)
            ...     return engine.all_to_all(x, axis='dp', split_axis=1, concat_axis=0)
        
        Use Cases:
            - Switching between parallelism strategies mid-computation
            - Attention: batch-sharded ↔ head-sharded
            - Expert routing in MoE models
        
        Note:
            Must be called within a parallel context.
            Only single axis supported (not tuple of axes).
        """
        return lax.all_to_all(
            arr,
            axis,
            split_axis=split_axis,
            concat_axis=concat_axis,
        )
    
    def ppermute(
        self,
        arr: jax.Array,
        axis: str,
        permutation: list[tuple[int, int]],
    ) -> jax.Array:
        """
        Permute array data between devices along a mesh axis.
        
        Routes arrays according to (src, dst) pairs. Each entry sends
        the array from device src to device dst along the specified axis.
        Devices not receiving data get zeros.
        
        Args:
            arr: Local array to route.
            axis: Mesh axis along which to permute.
            permutation: List of (source, destination) device index pairs.
                Indices are relative to the specified axis (0 to axis_size-1).
        
        Returns:
            jax.Array: Received array from routed source, or zeros.
        
        Diagram (ring shift right on 4-device 'tp' axis):
            permutation = [(0,1), (1,2), (2,3), (3,0)]
            
            Before:         After:
            tp=0: [A]       tp=0: [D]
            tp=1: [B]   →   tp=1: [A]
            tp=2: [C]       tp=2: [B]
            tp=3: [D]       tp=3: [C]
        
        Example:
            >>> # Ring shift along tensor-parallel axis
            >>> @engine.sharded(in_specs=P('dp', 'tp'), out_specs=P('dp', 'tp'))
            ... def ring_shift(x):
            ...     n = engine.axis_size('tp')
            ...     perm = [(i, (i + 1) % n) for i in range(n)]
            ...     return engine.ppermute(x, axis='tp', permutation=perm)
            
            >>> # Halo exchange for stencil operations
            >>> @engine.sharded(in_specs=P('dp'), out_specs=P('dp'))
            ... def get_left_neighbor(x):
            ...     n = engine.axis_size('dp')
            ...     # Each device receives from its left neighbor
            ...     perm = [((i + 1) % n, i) for i in range(n)]
            ...     return engine.ppermute(x, axis='dp', permutation=perm)
        
        Use Cases:
            - Ring-allreduce implementations
            - Halo exchanges for spatial parallelism
            - Pipeline stage communication
            - Butterfly/hypercube patterns
        
        Note:
            Must be called within a parallel context.
        """
        return lax.ppermute(arr, axis, permutation)
    
    def pbroadcast(
        self,
        arr: jax.Array,
        axis: str,
        source: int = 0,
    ) -> jax.Array:
        """
        Broadcast array from one device to all others along a mesh axis.
        
        Takes the array from source device and replicates it to all devices
        along the specified axis. Arrays on non-source devices are overwritten.
        
        Args:
            arr: Local array. Only source device's array is used.
            axis: Mesh axis along which to broadcast.
            source: Device index (0-indexed along the axis) to broadcast from.
        
        Returns:
            jax.Array: Source device's array, replicated along the axis.
        
        Diagram (broadcast from tp=0 on 4-device 'tp' axis):
            Before:         After:
            tp=0: [A]       tp=0: [A]
            tp=1: [B]   →   tp=1: [A]
            tp=2: [C]       tp=2: [A]
            tp=3: [D]       tp=3: [A]
        
        Example:
            >>> @engine.sharded(in_specs=P('dp', 'tp'), out_specs=P('dp', None))
            ... def broadcast_from_tp0(x):
            ...     return engine.pbroadcast(x, axis='tp', source=0)
        
        Note:
            Must be called within a parallel context.
        """
        return lax.pbroadcast(arr, axis, source=source)
    
    # =========================================================================
    # Local Device Queries (inside parallel contexts)
    # =========================================================================
    
    def local_device_id(self, axis: str) -> jax.Array:
        """
        Get device index along a specific mesh axis (inside parallel context).
        
        Returns the coordinate of this device along the specified axis.
        Must be called inside pmap, shard_map, or sharded jit.
        
        Args:
            axis: Mesh axis name to query.
        
        Returns:
            jax.Array: Scalar containing this device's index along the axis
                (0 to axis_size - 1).
        
        Example:
            >>> @engine.sharded(in_specs=P('dp', 'tp'), out_specs=P('dp', 'tp'))
            ... def get_coords():
            ...     dp_id = engine.local_device_id('dp')
            ...     tp_id = engine.local_device_id('tp')
            ...     return dp_id, tp_id
        """
        return lax.axis_index(axis)
    
    # =========================================================================
    # Sharding Utilities
    # =========================================================================
    
    def sharding(self, spec: P) -> NamedSharding:
        """
        Create a NamedSharding for array placement.
        
        Convenience method equivalent to:
            jax.NamedSharding(engine.mesh, spec)
        
        Args:
            spec: PartitionSpec using this engine's axis names.
                Use axis name for sharded dims, None for replicated.
        
        Returns:
            NamedSharding: Sharding object for device_put, jit args, or
                array creation with device= parameter.
        
        Example:
            >>> engine = JaxShardedEngine(
            ...     axis_shapes=(2, 4),
            ...     axis_names=('X', 'Y'),
            ... )
            >>> engine.set_global_mesh()
            >>> 
            >>> # Create arrays with explicit sharding
            >>> x = jnp.zeros((8, 1024), device=engine.sharding(P('X', 'Y')))
            >>> w = jnp.zeros((1024, 2048), device=engine.sharding(P(None, 'Y')))
            >>> 
            >>> # Place existing array
            >>> sharded_arr = jax.device_put(arr, engine.sharding(P('X')))
        """
        return NamedSharding(self._mesh, spec)
    
    def sharded(
        self,
        in_specs: P | tuple[P, ...],
        out_specs: P | tuple[P, ...],
        check_rep: bool = True,
    ) -> Callable:
        """
        Decorator to create a sharded SPMD function using shard_map.
        
        Wraps a function for execution across the device mesh. Inputs are
        partitioned per in_specs, the function runs on each partition,
        and outputs are assembled per out_specs.
        
        Args:
            in_specs: PartitionSpec(s) for inputs. Single spec or tuple
                matching function argument count.
            out_specs: PartitionSpec(s) for outputs.
            check_rep: Verify replicated outputs are identical across
                devices. Default: True
        
        Returns:
            Callable: Decorator transforming functions to sharded versions.
        
        Example:
            >>> engine = JaxShardedEngine(
            ...     axis_shapes=(2, 4),
            ...     axis_names=('dp', 'tp'),
            ... )
            >>> 
            >>> # Data-parallel forward with tensor-parallel weights
            >>> @engine.sharded(
            ...     in_specs=(P(None, 'tp'), P('dp', None)),
            ...     out_specs=P('dp', 'tp')
            ... )
            ... def matmul_sharded(w, x):
            ...     return x @ w
            >>> 
            >>> # Gradient sync across dp only
            >>> @engine.sharded(in_specs=P('dp', 'tp'), out_specs=P('dp', 'tp'))
            ... def sync_grads(g):
            ...     return engine.all_reduce(g, axis='dp', op='mean')
        
        PartitionSpec Patterns:
            - P(): Fully replicated
            - P('dp'): Sharded on axis 0 across dp
            - P('dp', 'tp'): Sharded on both axes
            - P(None, 'tp'): Replicated on axis 0, sharded on axis 1
            - P('dp', None, 'tp'): Mixed for 3D arrays
        
        Note:
            Inside the decorated function, arrays have *local* shapes
            (post-sharding). Collectives operate on these local views.
        """
        def decorator(fn: Callable) -> Callable:
            return shard_map(
                fn,
                mesh=self._mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=check_rep,
            )
        return decorator
    
    def shard_array(
        self,
        arr: npt.ArrayLike,
        spec: P,
    ) -> jax.Array:
        """
        Distribute an array across devices according to a partition spec.
        
        Takes a complete array and shards it across the mesh. The array
        shape must be compatible with the spec and mesh dimensions.
        
        Args:
            arr: Complete array to shard.
            spec: PartitionSpec defining the distribution.
        
        Returns:
            jax.Array: Sharded array distributed across devices.
        
        Example:
            >>> # Shard batch across dp, features across tp
            >>> data = jnp.ones((8, 1024))
            >>> sharded = engine.shard_array(data, P('dp', 'tp'))
            >>> # With 2×4 mesh: each device gets (4, 256)
        """
        return jax.device_put(arr, self.sharding(spec))
    
    def gather_array(self, arr: jax.Array) -> np.ndarray:
        """
        Collect a sharded array to host as numpy.
        
        Assembles all shards into a complete array on the calling process.
        
        Args:
            arr: Sharded jax.Array.
        
        Returns:
            np.ndarray: Fully assembled array.
        
        Warning:
            Triggers device-to-host transfer. Avoid in tight loops.
        """
        return np.asarray(arr)
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    def __enter__(self) -> 'JaxShardedEngine':
        """Enter mesh context for implicit sharding within block."""
        self._mesh.__enter__()
        return self
    
    def __exit__(self, *args) -> None:
        """Exit mesh context."""
        self._mesh.__exit__(*args)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def print_topology(self) -> None:
        """
        Print device topology information.
        
        Displays mesh shape, axis names, device counts, and platform.
        
        Example:
            >>> engine.print_topology()
            JAX Sharded Engine Topology
            ═══════════════════════════
            Process:         0 / 1
            Total Devices:   8
            Local Devices:   8
            Mesh Shape:      (2, 4)
            Mesh Axes:       ('dp', 'tp')
              dp: 2 devices
              tp: 4 devices
            Platform:        cuda
        """
        print("JAX Sharded Engine Topology")
        print("═" * 27)
        print(f"Process:         {self.device_id} / {self.num_processes}")
        print(f"Total Devices:   {self.num_devices}")
        print(f"Local Devices:   {self.num_local_devices}")
        print(f"Mesh Shape:      {self._axis_shapes}")
        print(f"Mesh Axes:       {self._axis_names}")
        for name, size in zip(self._axis_names, self._axis_shapes):
            print(f"  {name}: {size} devices")
        if self._devices:
            print(f"Platform:        {self._devices[0].platform}")
    
    def __repr__(self) -> str:
        axes_str = ", ".join(
            f"{n}={s}" for n, s in zip(self._axis_names, self._axis_shapes)
        )
        return (
            f"JaxShardedEngine({axes_str}, "
            f"process={self.device_id}/{self.num_processes})"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_data_parallel_engine(
    axis_name: str = 'dp',
    initialize: bool = False,
) -> JaxShardedEngine:
    """
    Create a 1D data-parallel engine across all devices.
    
    Args:
        axis_name: Name for the data-parallel axis.
        initialize: Initialize distributed runtime.
    
    Returns:
        JaxShardedEngine: 1D mesh for data parallelism.
    """
    num_devices = jax.device_count()
    return JaxShardedEngine(
        axis_shapes=(num_devices,),
        axis_names=(axis_name,),
        initialize_distributed=initialize,
    )


def create_2d_parallel_engine(
    dp_size: int,
    tp_size: int,
    dp_name: str = 'dp',
    tp_name: str = 'tp',
    initialize: bool = False,
) -> JaxShardedEngine:
    """
    Create a 2D mesh for combined data and tensor parallelism.
    
    Args:
        dp_size: Devices for data parallelism.
        tp_size: Devices for tensor parallelism.
        dp_name: Data parallel axis name.
        tp_name: Tensor parallel axis name.
        initialize: Initialize distributed runtime.
    
    Returns:
        JaxShardedEngine: 2D mesh (dp_size × tp_size).
    
    Example:
        >>> # 2 data-parallel × 4 tensor-parallel = 8 devices
        >>> engine = create_2d_parallel_engine(dp_size=2, tp_size=4)
    """
    return JaxShardedEngine(
        axis_shapes=(dp_size, tp_size),
        axis_names=(dp_name, tp_name),
        initialize_distributed=initialize,
    )