from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    hidden_size=5120,
    intermediate_size=20480,
    num_hidden_layers=64,
    num_attention_heads=32,
    num_key_value_heads=8,
    vocab_size=128000,
    max_position_embeddings=128000,
    attention_bias=False,
    bos_token_id=1,
    eos_token_id=2,
    rope_theta=10000.0,
)

model = LlamaForCausalLM(config)

# Model ve config kaydedilmeden önce validasyon
print(f"Model parametreleri: {model.num_parameters():,}")
print(f"Config kaydediliyor...")

model.save_pretrained("Alary21B")
config.save_pretrained("Alary21B")

print("✅ Model başarıyla kaydedildi!")
