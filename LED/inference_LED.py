import torch
import sys
import re
from tqdm import tqdm
from transformers import LEDTokenizer, LEDForConditionalGeneration
from peft import PeftModel

# 📌 Ensure a file path was passed
if len(sys.argv) != 2:
    print("Usage: python inference.py /path/to/document.txt")
    sys.exit(1)

# 🧹 Clean noisy text input
def clean_text(text):
    text = re.sub(r'▬+', ' ', text)
    text = re.sub(r'2\.\d+ Gradient Tree Boosting Programs', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 🧽 Clean repetitive summary phrases
def clean_summary(text):
    sentences = text.split('. ')
    seen = set()
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        if sent and sent.lower() not in seen:
            cleaned.append(sent)
            seen.add(sent.lower())
    return '. '.join(cleaned)

# 📄 Load and clean document
file_path = sys.argv[1]
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

text = clean_text(raw_text)

# 🔡 Load tokenizer and base model
base_model_name = "allenai/led-base-16384"
tokenizer = LEDTokenizer.from_pretrained(base_model_name)
base_model = LEDForConditionalGeneration.from_pretrained(base_model_name)

# 🔌 Load LoRA adapter
lora_adapter_path = "./led_lora_adapter"
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

# ✂️ Tokenize input and create global attention mask
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=4096,
)
inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
inputs["global_attention_mask"][:, 0] = 1

# ⚙️ Move model and input to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 🚀 Generate summary with updated decoding settings for more coherent output
print("🧠 Generating summary, please wait...")
with torch.no_grad():
    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        global_attention_mask=inputs["global_attention_mask"],
        max_new_tokens=350,
        do_sample=False,              # Use deterministic decoding for consistency
        num_beams=5,                  # Explore more candidate summaries
        no_repeat_ngram_size=3,       # Reduce repetitive phrases while permitting natural language flow
        repetition_penalty=1.2,       # Lower penalty to allow some natural repetition
        eos_token_id=tokenizer.eos_token_id,
    )

# 📜 Decode and display summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
summary = clean_summary(summary)

print("\n📄 Summary:\n", summary)
