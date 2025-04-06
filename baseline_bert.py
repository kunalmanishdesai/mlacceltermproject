import torch
import time
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizerFast
from datasets import load_metric
from torch.utils.data import DataLoader
import os
import wget
import json

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Load model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name).to(device).eval()
tokenizer = BertTokenizerFast.from_pretrained(model_name)


# Download and load SQuAD
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
    tokens["question"] = questions  # keep for printing
    tokens["context"] = contexts
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)

# Inference
latencies = []
all_preds, all_refs = [], []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        torch.cuda.synchronize()
        start = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        torch.cuda.synchronize()
        end = time.time()

        latencies.append(end - start)

        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()

        for i in range(len(input_ids)):
            start_idx = np.argmax(start_logits[i])
            end_idx = np.argmax(end_logits[i])
            input_tokens = input_ids[i].cpu().tolist()

            # Safeguard: invalid span
            if start_idx > end_idx:
                pred_ans = ""
            else:
                pred_ans = tokenizer.decode(input_tokens[start_idx:end_idx + 1], skip_special_tokens=True)

            all_preds.append({'id': batch["id"][i], 'prediction_text': pred_ans})
            all_refs.append({'id': batch["id"][i], 'answers': batch["answers"][i]})

# Print sample predictions
print("\n==== Sample Predictions ====")
for i in range(5):
    print(f"\nQ{i+1}: {all_refs[i]['id']}")
    print(f"Question: {dataset[i]['question']}")
    print(f"Context: {dataset[i]['context'][:100]}...")
    print(f"Predicted: {all_preds[i]['prediction_text']}")
    print(f"Actual: {all_refs[i]['answers'][0]['text']}")




# Metrics
results = metric.compute(predictions=all_preds, references=all_refs)
avg_latency = np.mean(latencies)

print(f"\n==== Baseline BERT Results ====")
print(f"Avg Latency per batch (batch_size=8): {avg_latency:.4f} sec")
print(f"F1 Score: {results['f1']:.2f}")
print(f"Exact Match (EM): {results['exact_match']:.2f}")
