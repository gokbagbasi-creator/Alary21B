"""
Alary21B - Instruction Tuning Training Script
Optimized for TPU Research Cloud
"""

import os
import torch
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset, Dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class InstructionTuningConfig:
    """Instruction Tuning Configuration"""
    model_id: str = "Alary21B_Final"
    dataset_name: str = "aya_dataset"
    output_dir: str = "instruction_tuning_checkpoints"
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # TPU Optimization
    use_tpu: bool = True
    max_seq_length: int = 2048
    
    # LoRA Configuration (optional)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class InstructionDataset:
    """Aya Dataset Processing"""
    
    @staticmethod
    def format_instruction(example: Dict) -> Dict:
        """Format instruction-following data"""
        if "instruction" in example:
            text = f"""### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
{example['output']}"""
        else:
            text = example.get('text', '')
        
        return {'text': text}
    
    @staticmethod
    def load_aya_dataset(config: InstructionTuningConfig) -> Dataset:
        """Load and process Aya dataset"""
        print("🔄 Aya dataset yükleniyor...")
        
        try:
            # Try to load from HF Hub
            dataset = load_dataset(config.dataset_name, split="train")
            print(f"✅ Dataset yüklendi: {len(dataset)} örnekler")
        except:
            # Fallback: local dataset
            if os.path.exists(config.dataset_name):
                print(f"📁 Lokal dataset'ten yükleniyor: {config.dataset_name}")
                with open(f"{config.dataset_name}/data.json", 'r') as f:
                    data = json.load(f)
                dataset = Dataset.from_dict(data)
            else:
                raise ValueError(f"Dataset bulunamadı: {config.dataset_name}")
        
        # Format dataset
        dataset = dataset.map(
            InstructionDataset.format_instruction,
            remove_columns=[col for col in dataset.column_names if col != 'text']
        )
        
        return dataset


class InstructionTuningTrainer:
    """Instruction Tuning Pipeline"""
    
    def __init__(self, config: InstructionTuningConfig):
        self.config = config
        self.device = "tpu" if config.use_tpu else "gpu"
        
    def setup_training(self):
        """Setup model, tokenizer, and training arguments"""
        print("🔧 Setup başlıyor...")
        
        # Load model and tokenizer
        print("📦 Model yükleniyor...")
        self.model = LlamaForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,  # Native TPU format
            device_map="auto" if not self.config.use_tpu else None,
        )
        
        print("🔤 Tokenizer yükleniyor...")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply LoRA if requested
        if self.config.use_lora:
            print("🎯 LoRA yapılandırılıyor...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Tokenization function
        def tokenize_function(examples):
            outputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                return_overflowing_tokens=False,
            )
            return outputs
        
        # Load and process dataset
        print("📊 Dataset işleniyor...")
        dataset = InstructionDataset.load_aya_dataset(self.config)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing..."
        )
        
        # Train/val split
        split_dataset = tokenized_dataset.train_test_split(test_size=0.05)
        
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]
        
        print(f"✅ Training set: {len(self.train_dataset)} örnekler")
        print(f"✅ Validation set: {len(self.eval_dataset)} örnekler")
        
        return self.model, self.tokenizer
    
    def train(self):
        """Start training"""
        print("🚀 Eğitim başlıyor...\n")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            
            # Logging
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            eval_strategy="steps",
            eval_steps=500,
            
            # TPU Optimization
            bf16=True,  # bfloat16 for TPU
            dataloader_pin_memory=False,
            optim="adafactor",  # Better for TPU
            gradient_checkpointing=True,
            
            # Misc
            seed=42,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        train_result = trainer.train()
        
        print("\n" + "="*50)
        print("✅ Eğitim Tamamlandı!")
        print(f"Final Loss: {train_result.training_loss}")
        print("="*50)
        
        # Save final model
        print("\n💾 Final model kaydediliyor...")
        self.model.save_pretrained(f"{self.config.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/final_model")
        
        print("✅ Model kaydedildi!")
        
        return trainer


def main():
    """Main training pipeline"""
    config = InstructionTuningConfig()
    
    print("="*60)
    print("🎯 ALARY21B - INSTRUCTION TUNING")
    print("="*60)
    print(f"⏰ Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Dataset: {config.dataset_name}")
    print(f"🔧 Device: {config.device}")
    print(f"💾 Output: {config.output_dir}")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = InstructionTuningTrainer(config)
    
    # Setup
    trainer.setup_training()
    
    # Train
    trainer.train()
    
    print(f"\n✨ Eğitim tamamlandı! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Çıktı: {config.output_dir}")


if __name__ == "__main__":
    main()
