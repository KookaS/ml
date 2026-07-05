import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size  # D
        self.num_attention_heads = config.num_attention_heads  # N
        self.num_key_value_heads = (
            config.num_key_value_heads
        )  # Grouped-Query Attention (GQA)
        self.head_dim = self.hidden_size // self.num_attention_heads  # H
        D, Nh, H, Nkv = (
            self.hidden_size,
            self.num_attention_heads,
            self.head_dim,
            self.num_key_value_heads,
        )

        self.q = nn.Linear(D, Nh * H, bias=False)
        self.k = nn.Linear(D, Nkv * H, bias=False)
        self.v = nn.Linear(D, Nkv * H, bias=False)
        self.o = nn.Linear(Nh * H, D, bias=False)

    def forward(self, x, position_embeddings, attention_mask):
        B, S, D = x.shape
        Nh, H, Nkv = (
            self.num_attention_heads,
            self.head_dim,
            self.num_key_value_heads,
        )

        cos, sin = position_embeddings

        wq = self.q.weight.view(Nh, H, D)
        wk = self.k.weight.view(Nkv, H, D)
        wv = self.v.weight.view(Nkv, H, D)
        wo = self.o.weight.view(D, Nh, H)

        # [B, N, S, H] layout is what HF's rope/repeat_kv expect
        q = torch.einsum("bsd,nhd->bnsh", x, wq)
        k = torch.einsum("btd,nhd->bnth", x, wk)
        v = torch.einsum("btd,nhd->bnth", x, wv)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # RoPE
        k = repeat_kv(k, Nh // Nkv)
        v = repeat_kv(v, Nh // Nkv)

        qk = torch.einsum("bnsh,bnth->bnst", q, k) / (self.head_dim**0.5)
        qk = qk + attention_mask  # MASK: [B, N, S, T] + [1, 1, S, T] where S = T
        qk = torch.softmax(qk, dim=-1)
        qkv = torch.einsum("bnst,bnth->bsnh", qk, v)
        out = torch.einsum("bsnh,dnh->bsd", qkv, wo)

        return out


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size  # D
        self.intermediate_size = config.intermediate_size  # Dff

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size  # D
        self.num_attention_heads = config.num_attention_heads  # N
        self.num_key_value_heads = (
            config.num_key_value_heads
        )  # Grouped-Query Attention (GQA)
        self.rms_norm_eps = config.rms_norm_eps

        self.pre_norm = LlamaRMSNorm(self.hidden_size, self.rms_norm_eps)
        self.attention = Attention(config)
        self.post_norm = LlamaRMSNorm(self.hidden_size, self.rms_norm_eps)
        self.mlp = Mlp(config)

    def forward(self, x, position_embeddings, attention_mask):
        # x --> norm + att --> x  --> norm + mlp --> x
        # |                   | |                    |
        # --------------------  ---------------------

        x_att = self.pre_norm(x)
        x_att = self.attention(x_att, position_embeddings, attention_mask)
        x = x + x_att  # residual connection

        x_mlp = self.post_norm(x)
        x_mlp = self.mlp(x_mlp)
        x = x + x_mlp  # residual connection

        return x


class Llama32(nn.Module):
    def __init__(self):
        super().__init__()

        self.config = LlamaConfig(
            vocab_size=128256,
            hidden_size=2048,  # D
            num_hidden_layers=16,
            num_attention_heads=32,  # N
            num_key_value_heads=8,  # Grouped-Query Attention (GQA)
            intermediate_size=8192,
            rms_norm_eps=1e-05,
            bos_token_id=128000,
            eos_token_id=128009,
        )

        self.rotary_embedding = LlamaRotaryEmbedding(config=self.config)

        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                TransformerDecoder(self.config, i)
                # LlamaDecoderLayer(self.config, i)
                for i in range(self.config.num_hidden_layers)
            ]
        )

        self.norm = LlamaRMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.head = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.vocab_size,
            bias=False,
        )

    def forward(self, input_tokens: torch.Tensor):

        # positional embeddings for each token with RoPE
        hidden_state = self.embedding(input_tokens)  # [B, S, D]
        _, S = input_tokens.shape
        device = hidden_state.device

        position_ids = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
        position_embeddings = self.rotary_embedding(
            hidden_state,
            position_ids,
        )  # (cos, sin)

        # decoder mask
        min_val = torch.finfo(hidden_state.dtype).min
        causal_mask = torch.full(
            (S, S),
            min_val,
            dtype=hidden_state.dtype,
            device=device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, S, S]

        for layer in self.layers:
            hidden_state = layer(
                hidden_state,
                # position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
            )  # [B, S, D]

        hidden_state = self.norm(hidden_state)
        logits = (
            hidden_state @ self.embedding.weight.T
        )  # reuse same embedding matrix, so need transpose
        return logits


def convert_key_custom(key: str) -> str:
    """Map a HuggingFace Llama key onto this model's naming scheme."""
    if key.startswith("model."):
        key = key.replace("model.", "", 1)

    key = key.replace("embed_tokens.weight", "embedding.weight")
    key = key.replace("input_layernorm", "pre_norm")
    key = key.replace("post_attention_layernorm", "post_norm")
    key = key.replace("self_attn.q_proj", "attention.q")
    key = key.replace("self_attn.k_proj", "attention.k")
    key = key.replace("self_attn.v_proj", "attention.v")
    key = key.replace("self_attn.o_proj", "attention.o")
    # mlp.gate_proj / mlp.up_proj / mlp.down_proj already match after stripping "model."
    return key


def convert_key_llama(key: str) -> str:
    if key.startswith("model."):
        key = key.replace("model.", "", 1)

    if key == "embed_tokens.weight":
        key = "embedding.weight"

    return key


def load_weights():
    weights_dir = "./weights"

    # Find all safetensors files in the directory
    safetensor_files = [
        os.path.join(weights_dir, f)
        for f in os.listdir(weights_dir)
        if f.endswith(".safetensors")
    ]

    if not safetensor_files:
        print(
            "source .env && "
            "uvx hf download meta-llama/Llama-3.2-1B-Instruct --local-dir ./weights"
        )
        exit(1)

    print(f"Found {len(safetensor_files)} weight shards. Initializing mapping...")

    model = Llama32()

    # Combine all shards into one dictionary
    hf_state_dict = {}
    for file_path in safetensor_files:
        hf_state_dict.update(load_file(file_path))

    # ---------------------------------------------------------
    # Inspect the keys
    # ---------------------------------------------------------
    # print("\n--- KEYS EXPECTED BY CUSTOM LLAMA32 MODEL ---")
    # expected_keys = sorted(list(model.state_dict().keys()))
    # for k in expected_keys[:]:
    #     print(f"  {k}")
    # print(f"  ... and {len(expected_keys) - 5} more keys.")

    # print("\n--- KEYS PRESENT IN HUGGING FACE SAFETENSORS ---")
    # hf_keys = sorted(list(hf_state_dict.keys()))
    # for k in hf_keys[:]:
    #     print(f"  {k}")

    # ---------------------------------------------------------

    clean_state_dict = {convert_key_custom(k): v for k, v in hf_state_dict.items()}
    # clean_state_dict = {convert_key_llama(k): v for k, v in hf_state_dict.items()}

    clean_state_dict["head.weight"] = clean_state_dict["embedding.weight"]  # [V, D]

    missing, unexpected = model.load_state_dict(clean_state_dict)
    if missing:
        print("Missing:", missing)
    if unexpected:
        print("Unexpected:", unexpected)
    if not missing and not unexpected:
        print("Pretrained parameters matched and loaded perfectly!")

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device}")

    # # -------------------------
    # # Default model loader
    # # -------------------------
    # print("Using HF pipeline...")
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"
    # pipe = pipeline(
    #     "text-generation",
    #     model=model_id,
    #     dtype=torch.bfloat16,
    #     device_map="auto",
    # )
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a pirate chatbot who always responds in pirate speak!",
    #     },
    #     {"role": "user", "content": "Who are you?"},
    # ]
    # outputs = pipe(
    #     messages,
    #     max_new_tokens=256,
    # )
    # print(outputs[0]["generated_text"][-1])

    # -------------------------
    # Custom model loader
    # -------------------------
    print("Using custom model loader...")
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    model = load_weights()
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        clean_up_tokenization_spaces=False,
    )

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a pirate chatbot who always responds in pirate speak!<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Who are you?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    input_token_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # [S, V]
    generated_tokens = []
    max_new_tokens = 30

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_token_ids)  # [Batch, Sequence, Vocab]
            next_token = logits[0, -1, :]
            next_token_id = torch.argmax(next_token, dim=-1, keepdim=True)  # [V]
            next_token_item = next_token_id.item()
            generated_tokens.append(next_token_item)

            if (
                next_token_item == tokenizer.eos_token_id or next_token_item == 128009
            ):  # <|eot_id|>
                break

            input_token_ids = torch.cat(
                [input_token_ids, next_token_id.unsqueeze(0)], dim=-1
            )

    output = tokenizer.decode(generated_tokens)
    print(output)
