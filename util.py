import jax
import time
from jax.sharding import NamedSharding
import copy

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

def benchmark(name, fn, *args, n_iter=10):
    # 1. Warmup
    # We pass a copy to avoid consuming the args during warmup
    print(f"Warming up {name}...")
    jax.block_until_ready(fn(*copy.deepcopy(args)))
    
    # 2. Pre-allocate inputs
    # We create N copies in memory *before* the clock starts.
    # This prevents the 'deepcopy' overhead from polluting your speed test.
    print(f"Preparing {n_iter} copies of inputs...")
    inputs = [copy.deepcopy(args) for _ in range(n_iter)]
    
    # 3. Loop
    start = time.time()
    for i in range(n_iter):
        # Use one of the pre-made copies
        # jax.block_until_ready handles tuples/dicts automatically
        jax.block_until_ready(fn(*inputs[i]))
    end = time.time()
    
    # 4. Report
    avg_time = (end - start) / n_iter
    print(f"{name}: {avg_time:.6f} s/iter")

    # print_memory_stats("Before Run")
    # jax.block_until_ready(fn(*args))
    # print_memory_stats("After Run")

def print_memory_stats(step_name=""):
    print(f"\n--- Memory Stats [{step_name}] ---")
    # Get the first local device (assuming roughly symmetric usage across chips)
    device = jax.local_devices()[0]
    
    # This returns a dictionary like {'bytes_in_use': 1024, 'peak_bytes_in_use': 2048}
    stats = device.memory_stats()
    
    if not stats:
        print("  Memory stats not available (Common on CPU simulation)")
        return

    # Convert to MB/GB
    in_use_mb = stats.get('bytes_in_use', 0) / 1024**2
    peak_mb = stats.get('peak_bytes_in_use', 0) / 1024**2
    
    print(f"  Current Usage: {in_use_mb:.2f} MB")
    print(f"  Peak Usage:    {peak_mb:.2f} MB")

@jax.jit
def relu(x):
    return x * (x > 0)