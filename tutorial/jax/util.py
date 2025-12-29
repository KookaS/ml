import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

def inspect_array(arr, name="Array"):
    print(f"\n--- Inspecting: {name} ---")
    print(f"  Global Shape: {arr.shape}")
    print(f"  DType:        {arr.dtype}")
    
    # Check if it has sharding info
    if not hasattr(arr, 'sharding') or not isinstance(arr.sharding, NamedSharding):
        print("  Sharding:     None (Default/Replicated)")
        return

    spec = arr.sharding.spec
    mesh = arr.sharding.mesh
    print(f"  Sharding Spec: {spec}")
    
    # Iterate over each dimension to show splitting details
    for i, (dim_name, dim_size) in enumerate(zip(spec, arr.shape)):
        if dim_name is None:
            print(f"  - Axis {i} (Size {dim_size}): Replicated (Full copy on all devices)")
        else:
            # Calculate local chunk size
            mesh_dim_size = mesh.shape[dim_name]
            local_size = dim_size // mesh_dim_size
            print(f"  - Axis {i} (Size {dim_size}): Sharded on '{dim_name}' "
                  f"| Mesh Size: {mesh_dim_size} | Local Chunk: {local_size}")