from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


tokenizer = Tokenizer(BPE())

trainer = BpeTrainer(
    vocab_size=128000,
    special_tokens=[
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>"
    ]
)


ds_list = [
    load_dataset("uonlp/CulturaX", streaming=True, split="train", trust_remote_code=True),
    load_dataset("bigcode/starcoderdata", streaming=True, split="train", trust_remote_code=True),
    load_dataset("bigcode/the-stack-v2", streaming=True, split="train", trust_remote_code=True)
]

dataset = interleave_datasets(ds_list, probabilities=[0.8, 0.1, 0.1])


def batch_iterator(batch_size=1000):
    """Veri setinden text çıkarmak için iterator."""
    batch = []
    for example in dataset:
        # Farklı veri setlerinde 'text' field'i farklı isimlerde olabilir
        text = None
        if "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        elif "code" in example:
            text = example["code"]
        
        if text and isinstance(text, str):
            batch.append(text)
            
            if len(batch) == batch_size:
                yield batch
                batch = []
    
    # Kalan batch'i gönder
    if batch:
        yield batch


print("Tokenizer eğitiliyor...")
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer
)

print("✅ Tokenizer eğitimi tamamlandı!")
tokenizer.save("tokenizer.json")
print("✅ Tokenizer kaydedildi!")
