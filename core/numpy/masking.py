import numpy as np

# 1D MASKING
arr = np.random.normal(size=(1024, 256))
padding = 30

mask = np.arange(arr.shape[0]) <= arr.shape[0] - padding # (1024,)
mask = mask[:, None] # (1024, 1)

masked_arr = arr * mask # (1024, 256) * (1024, 1)

# 2D MASKING (lower triangular)
size = 4
mask = np.arange(size)[:, None] >= np.arange(size)[None, :] # lower triangular matrix and diagonals filled with True
mask = np.where(mask, 0, -np.inf) # lower and diagonals 0, upper -inf

# BOOLEAN ARRAY INDEXING
arr = np.random.normal(size=(10,))
mask = arr > 0
positives = arr[mask]
print(f"{mask=}")
print(f"{positives=}")