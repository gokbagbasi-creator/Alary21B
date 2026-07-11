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


def batch_iterator():
    for example in dataset:
        if "text" in example:
            yield example["text"]


tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer
)


tokenizer.save("tokenizer.json")
