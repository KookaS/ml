from typing import Sequence
import torch

from tutorial.torch.softmax import softmax

class KVCache:
    def __init__(self, batch_size, max_sequence_length, num_heads, head_dim):
        """
        Creation of the KV cache.
        Shape: (S, D)
        """
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads
        self.head_dim = head_dim

        # pre-allocation, Paged-Attention improves this
        self.pointer = 0
        self.k_cache = torch.empty((batch_size, max_sequence_length, num_heads, head_dim))
        self.v_cache = torch.empty((batch_size, max_sequence_length, num_heads, head_dim))

    def prefill(self, k, v):
        """
        Fill the kv cache.
        Shape: (B, S, N, H)
        """
        B, S, N, H = k.shape

        if B != self.batch_size:
            raise ValueError(f"key and value should have a batch size of {self.batch_size}")
        
        self.k_cache[:, :S, :, :] = k
        self.v_cache[:, :S, :, :] = v
        self.pointer = S + 1
    
    def decode(self, k_new, v_new):
        """
        Add a token to the KVCache.
        Shape: (B, 1, N, H)
        """
        if k_new.shape[1] != 1:
            raise ValueError("key and value should have only one token")
        
        if self.pointer >= self.max_sequence_length:
            raise IndexError("KV Cache is full")
        
        self.k_cache[:, self.pointer, :, :] = k_new.squeeze(1)
        self.v_cache[:, self.pointer, :, :] = v_new.squeeze(1)
        self.pointer += 1

    def get(self, batch_size: int = None):
        """
        Get the KVCache.
        """
        view_batch_size = batch_size or self.batch_size
        return (
            self.k_cache[:view_batch_size, :self.pointer, :, :],
            self.v_cache[:view_batch_size, :self.pointer, :, :],
        )
    
    @staticmethod
    def fuse(*caches: "KVCache"):
        """
        Fuse the KVCache together of the same sequence length.
        """
        # [B, S, N, H]
        batch_size = len(caches)
        max_sequence_length = caches[0].max_sequence_length
        num_heads = caches[0].num_heads
        head_dim = caches[0].head_dim
        kv_cache_fused = KVCache(batch_size, max_sequence_length, num_heads, head_dim)

        for i, cache in enumerate(caches):
            k, v = cache.get() # [1, S, N, H]
            if k.shape[0] != 1:
                raise ValueError("Key and Value should have batch size of 1")
            sequence_length = k.shape[1]
            kv_cache_fused.k_cache[i, :sequence_length, :, :] = k.squeeze(0)
            kv_cache_fused.v_cache[i, :sequence_length, :, :] = v.squeeze(0)
        kv_cache_fused.pointer = caches[0].pointer
        
        return kv_cache_fused

class _AttentionDisaggregatedFn:
    """
    Multi head attention with disaggregated serving for inference, with Prefill and Decode.

    B batch size
    S query size
    T key/value size
    N number of heads
    H head dimension
    """

    @staticmethod
    def prefill(x, wq, wk, wv, wo, kv_cache):
        """
        During Prefill, the kv_cache is built in one pass, where all tokens of the prompt attend to each other. Since we process an entire sequence, we want to have a low batch size.
        """
        q = torch.einsum('bsd,dnh->bsnh', x, wq)
        k = torch.einsum('btd,dnh->btnh', x, wk)
        v = torch.einsum('btd,dnh->btnh', x, wv)

        # KV CACHE saving for Decode
        kv_cache.prefill(k, v)

        # SCALING
        qk = torch.einsum('bsnh,btnh->bsnt', q, k) # contract over head dimensions
        qk /= q.shape[-1]**0.5

        # MASKING (only prefill)
        seq = qk.shape[1]
        mask = torch.arange(seq)[:, None] >= torch.arange(seq)[None, :] 
        mask = torch.where(mask, 0.0, -torch.inf)
        qk += mask[None, :, None, :] # B S N T

        # ATTENTION SCORE
        scores = softmax(qk, dim=-1) # the -inf will turn to 0 with softmax

        qkv = torch.einsum('bsnt,btnh->bsnh', scores, v)
        
        attention = torch.einsum('bsnh,dnh->bsd', qkv, wo)

        return attention
    
    @staticmethod
    def decode(x_new, wq, wk, wv, wo, kv_cache):
        """
        During Decode, we process one token at a time, therefore we want a large batch size.
        """
        # only process the last token, S=1
        q = torch.einsum('bsd,dnh->bsnh', x_new, wq)
        k = torch.einsum('bsd,dnh->bsnh', x_new, wk)
        v = torch.einsum('bsd,dnh->bsnh', x_new, wv)

        # KV CACHE Updating with the latest token
        kv_cache.decode(k, v)
        k,v = kv_cache.get()

        # SCALING
        qk = torch.einsum('bsnh,btnh->bsnt', q, k) # contract over head dimensions
        qk /= q.shape[-1]**0.5

        # NO MASKING

        # ATTENTION SCORE
        scores = softmax(qk, dim=-1) # the -inf will turn to 0 with softmax

        qkv = torch.einsum('bsnt,btnh->bsnh', scores, v)
        
        attention = torch.einsum('bsnh,dnh->bsd', qkv, wo)

        return attention
    
class AttentionDisaggregated(torch.nn.Module):
    def __init__(self, d_model, n_heads, head_dim):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim

        # init the weights for the optimizer
        self.wq = torch.nn.Parameter(torch.empty(d_model, n_heads, head_dim))
        self.wk = torch.nn.Parameter(torch.empty(d_model, n_heads, head_dim))
        self.wv = torch.nn.Parameter(torch.empty(d_model, n_heads, head_dim))
        self.wo = torch.nn.Parameter(torch.empty(d_model, n_heads, head_dim))

        # xavier noise to keep the variance small, for controlled gradients
        torch.nn.init.xavier_normal_(self.wq)
        torch.nn.init.xavier_normal_(self.wk)
        torch.nn.init.xavier_normal_(self.wv)
        torch.nn.init.xavier_normal_(self.wo)
    
    def load_checkpoint(self, params):
        raise NotImplementedError

    def prefill(self, x, kv_cache):
        return _AttentionDisaggregatedFn.prefill(x, self.wq, self.wk, self.wv, self.wo, kv_cache)
    
    def decode(self, x, kv_cache):
        return _AttentionDisaggregatedFn.decode(x, self.wq, self.wk, self.wv, self.wo, kv_cache)    


def inference_loop(x: Sequence[torch.Tensor], max_sequence_length: int, d_model: int, n_heads: int, head_dim: int):
    """
    Dummy inference loop to mimic disaggregated serving.

    Schema:
    Req1 --> Cluster_prefill_1 \
                                \
                                    --> Cluster_decode --> fuse --> decode
                                /
    Req2 --> Cluster_prefill_2 /
    
    There are two types of load balancers, with two types of clusters, each with their own dimensions and sharding strategy.

    :param x: list of prompts, each of shape [1, S, D]
    """
    # We assume all requests have the same size (or are padded)
    B = len(x)
    _, S, D = x[0].shape
    print(f"[System] Incoming requests: {B}, Sequence Length: {S}, Model Dim: {D}")
    if S > max_sequence_length:
        raise ValueError(f"Sequence length too long for context size {max_sequence_length}")

    # each prefill and decode should have their own clusters with sharding strategies
    model = AttentionDisaggregated(d_model, n_heads, head_dim)
    # model.load_checkpoint()

    # PREFILL
    # KVCache with batch of 1
    batch_size_prefill = 1
    requests = [] # mimic the requests between prefill and decode
    # this loop mimic the load balancer work
    for i in range(B):
        # model should be sharded with Tensor Parallelism(Megatron) for faster compute
        kv_cache_prefill = KVCache(batch_size_prefill, max_sequence_length, n_heads, head_dim)
        attention = model.prefill(x[i], kv_cache_prefill)
        # .... --> x_new
        x_new_prefill = torch.randn((batch_size_prefill, 1, D))
        # send kv_cache and x_new to another cluster made for decode
        requests.append((kv_cache_prefill, x_new_prefill))
        print(f"[Cluster Prefill {i}] Cache Size: {kv_cache_prefill.pointer}")
    print(f"[Network] Sent {len(requests)} caches of size [{requests[0][0].pointer}] to [Cluster Decode 0]")

    # KV CACHE FUSING
    # we mimic that all prefill requests are done simultaneously
    batch_size_decode = len(requests)
    kv_caches = []
    x_new_batched = torch.empty(batch_size_decode, 1, D)
    for i, (kv_cache_prefill, x_new_prefill) in enumerate(requests):
        # aggreagete all kv_caches received together
        kv_caches.append(kv_cache_prefill)
        # aggregate all new tokens, [1, 1, D] -> [B, 1, D]
        x_new_batched[i, :, :] = x_new_prefill.squeeze(0)
    del requests
    kv_cache_decode = KVCache.fuse(*kv_caches)
    del kv_caches
    print(f"[Cluster Decode 0] Fused {batch_size_decode} caches, New Batch Size: {kv_cache_decode.batch_size}, Cache Pointer: {kv_cache_decode.pointer}")

    # DECODE        
    # loop until the last token is <EOS>
    for i in range(3): # dummy loop
        # model should be sharded with DP or FSDP for maximizing throughput
        attention = model.decode(x_new_batched, kv_cache_decode)
        # .... --> x_new
        x_new_batched = torch.randn_like(x_new_batched)
        print(f"[Cluster Decode 0] Step {i+1}: Cache Pointer: {kv_cache_decode.pointer}")

if __name__ == "__main__":
    D = 16
    N = 4
    H = 4
    
    # Simulate 2 user requests with exact same sequence length
    req1 = torch.randn(1, 10, D)
    req2 = torch.randn(1, 10, D)
    
    max_sequence_length = 100
    inference_loop([req1, req2], max_sequence_length, D, N, H)