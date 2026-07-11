from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    hidden_size=5120,
    intermediate_size=20480,
    num_hidden_layers=64,
    num_attention_heads=32,
    num_key_value_heads=8,
    vocab_size=128000,
    max_position_embeddings=128000,
    attention_bias=False
)

model = LlamaForCausalLM(config)

model.save_pretrained("Alary21B")
config.save_pretrained("Alary21B")
