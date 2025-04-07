
import torch
import time
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizerFast
from datasets import load_metric
from torch.utils.data import DataLoader
import os
import wget
import json
import tempfile

# Force CPU for quantization
device = torch.device("cpu")
print(f"Running on device: {device}")

# Load fine-tuned BERT model
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Apply dynamic quantization (linear layers only)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
).to(device).eval()

# Show quantized layers
print("\n=== Quantized Model Structure ===")
print(quantized_model)

# Compare model sizes
def get_model_size(model):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / 1e6
        os.unlink(f.name)
        return size_mb

original_size = get_model_size(model)
quantized_size = get_model_size(quantized_model)

print(f"\nOriginal BERT size: {original_size:.2f} MB")
print(f"Quantized BERT size: {quantized_size:.2f} MB")

# Load SQuAD data
cache_dir = "./squad_data"
os.makedirs(cache_dir, exist_ok=True)
squad_path = os.path.join(cache_dir, "dev-v1.1.json")

if not os.path.exists(squad_path):
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    print("Downloading SQuAD...")
    wget.download(url, squad_path)

def load_squad_json(path, limit=None):
    with open(path, "r") as f:
        raw_data = json.load(f)

    examples = []
    for article in raw_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                examples.append({
                    "id": qa["id"],
                    "context": context,
                    "question": qa["question"],
                    "answers": qa["answers"]
                })
                if limit and len(examples) >= limit:
                    return examples
    return examples

dataset = load_squad_json(squad_path, limit=128)
metric = load_metric("squad")

# DataLoader
def collate(batch):
    questions = [x["question"] for x in batch]
    contexts = [x["context"] for x in batch]
    ids = [x["id"] for x in batch]
    answers = [x["answers"] for x in batch]

    tokens = tokenizer(
        questions,
        contexts,
        padding=True,
        truncation=True,
        max_length=384,
        return_tensors="pt"
    )
    tokens["id"] = ids
    tokens["answers"] = answers
    tokens["question"] = questions
    tokens["context"] = contexts
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)

# Inference loop
latencies = []
all_preds, all_refs = [], []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        start = time.time()
        outputs = quantized_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        end = time.time()

        latencies.append(end - start)

        start_logits = outputs.start_logits.numpy()
        end_logits = outputs.end_logits.numpy()

        for i in range(len(input_ids)):
            start_idx = np.argmax(start_logits[i])
            end_idx = np.argmax(end_logits[i])
            input_tokens = input_ids[i].tolist()

            if start_idx > end_idx:
                pred_ans = ""
            else:
                pred_ans = tokenizer.decode(input_tokens[start_idx:end_idx + 1], skip_special_tokens=True)

            all_preds.append({'id': batch["id"][i], 'prediction_text': pred_ans})
            all_refs.append({'id': batch["id"][i], 'answers': batch["answers"][i]})

# Sample predictions
print("\n==== Sample Predictions (Quantized BERT) ====")
for i in range(5):
    print(f"\nQ{i+1}: {dataset[i]['question']}")
    print(f"Context: {dataset[i]['context'][:100]}...")
    print(f"Predicted: {all_preds[i]['prediction_text']}")
    print(f"Actual: {all_refs[i]['answers'][0]['text']}")

# Evaluate
results = metric.compute(predictions=all_preds, references=all_refs)
avg_latency = np.mean(latencies)

print(f"\n==== Quantized BERT Results ====")
print(f"Avg Latency per batch (CPU, batch_size=8): {avg_latency:.4f} sec")
print(f"F1 Score: {results['f1']:.2f}")
print(f"Exact Match (EM): {results['exact_match']:.2f}")
