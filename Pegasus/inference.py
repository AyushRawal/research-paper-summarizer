######## Use this file to perform inference without starting a web server ###########
#### takes text to be summarized as command line argument or reads `input.txt` ####

# Install required packages (run once)
# pip install sentence-transformers transformers nltk torch tqdm

import sys
import nltk
import torch
from nltk.tokenize import sent_tokenize
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from tqdm import tqdm

# Download NLTK data (for sentence tokenization)
nltk.download("punkt")

# ----------- CONFIG -----------
MODEL_DIR = "pegasus/pegasus_arxiv_model/final_subset"  # Your fine-tuned Pegasus model
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
INPUT_TOKEN_LIMIT = 1024            # Max tokens per chunk for Pegasus input
SUMMARY_TOKEN_LIMIT = 400           # Max tokens per summary
FINAL_THRESHOLD = 800               # Recursively summarize if output > this token count

# ----------- LOAD MODELS -----------
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = PegasusTokenizer.from_pretrained(MODEL_DIR)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

# ----------- UTILITY FUNCTIONS -----------

def split_text_into_chunks(text, tokenizer, token_limit):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_tokens = [], "", 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, truncation=False)
        token_count = len(sentence_tokens)

        if token_count > token_limit:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk, current_tokens = "", 0
            chunks.append(sentence)
            continue

        if current_tokens + token_count > token_limit:
            chunks.append(current_chunk.strip())
            current_chunk, current_tokens = sentence, token_count
        else:
            current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
            current_tokens += token_count

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_text(text, model, tokenizer, input_token_limit, summary_token_limit):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="longest",
        max_length=input_token_limit,
        return_tensors="pt"
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=summary_token_limit,
        min_length=100,
        num_beams=5,
        early_stopping=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        length_penalty=1.0
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def recursive_summarize(text, model, tokenizer, input_token_limit, summary_token_limit, final_threshold):
    chunks = split_text_into_chunks(text, tokenizer, input_token_limit)
    print(f"\n📦 Splitting into {len(chunks)} chunk(s). Summarizing each...\n")

    chunk_summaries = []
    for chunk in tqdm(chunks, desc="Summarizing chunks"):
        summary = summarize_text(chunk, model, tokenizer, input_token_limit, summary_token_limit)
        chunk_summaries.append(summary)

    combined_summary = " ".join(chunk_summaries)

    # ✨ Add structured summary prompt
    structured_prompt = (
        "Summarize the following scientific text using the format:\n"
        "Motivation:\nMethod:\nResults:\nConclusion:\n\n"
        + combined_summary
    )

    print("\n🧠 Final pass with structured summary prompt...\n")
    final_summary = summarize_text(structured_prompt, model, tokenizer, input_token_limit, summary_token_limit)

    if len(tokenizer.encode(final_summary, truncation=False)) > final_threshold:
        print("\n🔁 Final summary too long. Recursing...\n")
        return recursive_summarize(final_summary, model, tokenizer, input_token_limit, summary_token_limit, final_threshold)
    else:
        return final_summary

# ----------- LOAD INPUT TEXT -----------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    article = f.read().strip()

print("\n🔍 Original Article (first 500 characters):\n")
print(article[:500] + "\n...")

# ----------- RUN STRUCTURED SUMMARIZATION -----------
final_summary = recursive_summarize(
    article, model, tokenizer,
    input_token_limit=INPUT_TOKEN_LIMIT,
    summary_token_limit=SUMMARY_TOKEN_LIMIT,
    final_threshold=FINAL_THRESHOLD
)

print("\n📚 Structured Final Summary:\n")
print(final_summary)
print("\n🧠 Final summary token count (approx):", len(tokenizer.encode(final_summary, truncation=False)))
