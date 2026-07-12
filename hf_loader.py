from transformers import LlamaForCausalLM, AutoTokenizer
from tokenizers import Tokenizer

REPO_ID = "Gokturk97/Alary21B"

try:
    print("Model yükleniyor...")
    model = LlamaForCausalLM.from_pretrained("Alary21B_Final")
    
    print("Tokenizer yükleniyor...")
    # Tokenizer dosyasından yükle
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except:
        # Eğer tokenizer.json yoksa, model dizininden yükle
        tokenizer = AutoTokenizer.from_pretrained("Alary21B_Final")
    
    print("Model Hugging Face Hub'a yükleniyor...")
    model.push_to_hub(REPO_ID)
    
    print("Tokenizer Hugging Face Hub'a yükleniyor...")
    if isinstance(tokenizer, Tokenizer):
        # tokenizers.Tokenizer nesnesi ise, önce dönüştür
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    tokenizer.push_to_hub(REPO_ID)
    
    print("✅ Upload tamamlandı!")
    
except Exception as e:
    print(f"❌ Hata oluştu: {e}")
    raise
