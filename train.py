import os, shutil, torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import LlamaForCausalLM, Adafactor
from torch.utils.data import DataLoader
from dataset import get_dataset, collate_fn

set_seed(42)
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=8)
output_dir = "checkpoints"
os.makedirs(output_dir, exist_ok=True)

model = LlamaForCausalLM.from_pretrained("Alary21B")
model.gradient_checkpointing_enable()

optimizer = Adafactor(model.parameters(), lr=1e-4, relative_step=False, scale_parameter=False)

train_dl = DataLoader(get_dataset("train"), batch_size=4, collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_dl = DataLoader(get_dataset("validation"), batch_size=4, collate_fn=collate_fn, num_workers=2, pin_memory=True)

model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

# Resume logic
last_ckpt = sorted([os.path.join(output_dir, d) for d in os.listdir(output_dir) if "checkpoint-" in d])
if last_ckpt:
    accelerator.load_state(last_ckpt[-1])
    print(f"Resumed from {last_ckpt[-1]}")

def evaluate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dl:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
            total_loss += outputs.loss.item()
    print(f"Validation Loss: {total_loss / len(val_dl)}")
    model.train()

model.train()
for step, batch in enumerate(train_dl):
    with accelerator.accumulate(model):
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
        accelerator.backward(outputs.loss)
        optimizer.step()
        optimizer.zero_grad()
    
    if step > 0 and step % 1000 == 0:
        accelerator.save_state(os.path.join(output_dir, f"checkpoint-{step}"))
        evaluate()
        
    if step % 100 == 0 and accelerator.is_main_process:
        print(f"Step: {step}, Loss: {outputs.loss.item()}")

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.unwrap_model(model).save_pretrained("Alary21B_Final")

