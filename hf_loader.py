from transformers import LlamaForCausalLM, AutoTokenizer

REPO_ID = "Gokturk97/Alary21B"

print("Model yükleniyor...")
model = LlamaForCausalLM.from_pretrained("Alary21B_Final")

print("Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("Alary21B_Final")

print("Model Hugging Face Hub'a yükleniyor...")
model.push_to_hub(REPO_ID)

print("Tokenizer Hugging Face Hub'a yükleniyor...")
tokenizer.push_to_hub(REPO_ID)

print("✅ Upload tamamlandı!")
