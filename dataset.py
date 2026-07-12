from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
import torch

tokenizer = Tokenizer.from_file("tokenizer.json")
tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
tokenizer.enable_truncation(max_length=4096)

def tokenize_function(examples):
    encoded = tokenizer.encode_batch(examples["text"])
    return {"input_ids": [e.ids for e in encoded], "attention_mask": [e.attention_mask for e in encoded]}

def get_dataset(split="train", probabilities=[0.8, 0.1, 0.1]):
    ds_list = [
        load_dataset("uonlp/CulturaX", streaming=True, split=split, trust_remote_code=True),
        load_dataset("bigcode/starcoderdata", streaming=True, split=split, trust_remote_code=True),
        load_dataset("bigcode/the-stack-v2", streaming=True, split=split, trust_remote_code=True)
    ]
    # Probabilities toplamı 1.0 olmalı
    total_prob = sum(probabilities)
    probabilities = [p / total_prob for p in probabilities]
    return interleave_datasets(ds_list, probabilities=probabilities).map(tokenize_function, batched=True)

def collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch])
    }
