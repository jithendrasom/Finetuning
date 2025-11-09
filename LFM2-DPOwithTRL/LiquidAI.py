# !pip install transformers==4.54.0 trl>=0.18.2 peft>=0.15.2

import torch
import transformers
import trl
import os
os.environ["WANDB_DISABLED"] = "true"

print(f"📦 PyTorch version: {torch.__version__}")
print(f"🤗 Transformers version: {transformers.__version__}")
print(f"📊 TRL version: {trl.__version__}")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "LiquidAI/LFM2-1.2B" # <- change model here to use LiquidAI/LFM2-700M or LiquidAI/LFM2-350M

print("📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("🧠 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)

print("✅ Local model loaded successfully!")
print(f"🔢 Parameters: {model.num_parameters():,}")
print(f"📖 Vocab size: {len(tokenizer)}")
print(f"💾 Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")

from datasets import load_dataset

print("📥 Loading DPO dataset...")

dataset_dpo = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train[:2000]")
dataset_dpo = dataset_dpo.train_test_split(test_size=0.1, seed=42)
train_dataset_dpo, eval_dataset_dpo = dataset_dpo['train'], dataset_dpo['test']

print("✅ DPO Dataset loaded:")
print(f"   📚 Train samples: {len(train_dataset_dpo)}")
print(f"   🧪 Eval samples: {len(eval_dataset_dpo)}")

sample = train_dataset_dpo[0]
print("\n📝 Single Sample:")
print(f"   Prompt: {sample['prompt'][:100]}...")
print(f"   ✅ Chosen: {sample['chosen'][:100]}...")
print(f"   ❌ Rejected: {sample['rejected'][:100]}...")


from peft import LoraConfig, get_peft_model, TaskType

GLU_MODULES = ["w1", "w2", "w3"]
MHA_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
CONV_MODULES = ["in_proj", "out_proj"]

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # <- lower values = fewer parameters
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=GLU_MODULES + MHA_MODULES + CONV_MODULES,
    bias="none",
    modules_to_save=None,
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

print("✅ LoRA configuration applied!")
print(f"🎛️  LoRA rank: {lora_config.r}")
print(f"📊 LoRA alpha: {lora_config.lora_alpha}")
print(f"🎯 Target modules: {lora_config.target_modules}")

## LAUNCH TRAINING

from trl import DPOConfig, DPOTrainer

# DPO Training configuration
dpo_config = DPOConfig(
    output_dir="./lfm2-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=1e-6,
    lr_scheduler_type="linear",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=False # <- not all colab GPUs support bf16
)

# Create DPO trainer
print("🏗️  Creating DPO trainer...")
dpo_trainer = DPOTrainer(
    model=lora_model,
    args=dpo_config,
    train_dataset=train_dataset_dpo,
    eval_dataset=eval_dataset_dpo,
    processing_class=tokenizer,
)

# Start DPO training
print("\n🚀 Starting DPO training...")
dpo_trainer.train()

print("🎉 DPO training completed!")

## MERGE BACK
# Save the DPO model
dpo_trainer.save_model()
print(f"💾 DPO model saved to: {dpo_config.output_dir}")

print("\n🔄 Merging LoRA weights...")
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("./lfm2-lora-merged")
tokenizer.save_pretrained("./lfm2-lora-merged")
print("💾 Merged model saved to: ./lfm2-lora-merged")
