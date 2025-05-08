# Step 1: Install required libraries
!pip install -q transformers datasets peft accelerate

# Step 2: Import modules and check for GPU availability
import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

# Step 3: Set up the LED model, tokenizer, and LoRA configuration
model_name = "allenai/led-base-16384"
tokenizer = LEDTokenizer.from_pretrained(model_name)
base_model = LEDForConditionalGeneration.from_pretrained(model_name)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# Step 4: Define tokenization parameters and preprocessing function
max_input_length = 4096
max_target_length = 512

def preprocess_function(batch):
    # Tokenize the article (input)
    inputs = tokenizer(
        batch["article"],
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
    )
    # Tokenize the abstract (target)
    targets = tokenizer(
        batch["abstract"],
        truncation=True,
        padding="max_length",
        max_length=max_target_length,
    )
    # Set labels for the model
    inputs["labels"] = targets["input_ids"]
    # Add global attention mask for LED (1 for first token, 0 for the rest)
    inputs["global_attention_mask"] = [
        [1] + [0] * (max_input_length - 1) for _ in range(len(inputs["input_ids"]))
    ]
    return inputs

# Step 5: Load (or tokenize & save) train and validation splits
if os.path.exists("tokenized_arxiv_train") and os.path.exists("tokenized_arxiv_val"):
    print("Loading tokenized datasets from session storage...")
    tokenized_train = load_from_disk("tokenized_arxiv_train")
    tokenized_val = load_from_disk("tokenized_arxiv_val")
else:
    print("Downloading and tokenizing dataset...")
    dataset = load_dataset("ccdv/arxiv-summarization")
    tokenized_train = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=["article", "abstract"]
    )
    tokenized_val = dataset["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=["article", "abstract"]
    )
    tokenized_train.save_to_disk("tokenized_arxiv_train")
    tokenized_val.save_to_disk("tokenized_arxiv_val")

# To reduce memory usage on Colab, use a subset for training
train_subset = tokenized_train.select(range(1000))
val_subset = tokenized_val.select(range(500))

# Step 6: Setup the data collator and TrainingArguments
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="/content/led_lora_arxiv_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,             # Use 1 epoch for testing; increase as needed
    save_total_limit=1,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,    # Reduces memory usage
    report_to="none",               # Disables logging to WandB unless desired
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=val_subset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 7: Fine-tune the model
print("🚀 Starting LoRA fine-tuning on LED...")
trainer.train()

# Step 8: Save the LoRA adapter to session storage
save_path = "/content/led_lora_adapter"
model.save_pretrained(save_path)
print("✅ LoRA adapter saved to:", save_path)

# Step 9: Zip the adapter folder and download it
!zip -r led_lora_adapter.zip /content/led_lora_adapter

from google.colab import files
files.download("led_lora_adapter.zip")
