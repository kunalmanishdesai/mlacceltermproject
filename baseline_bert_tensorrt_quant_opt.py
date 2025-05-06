import torch
import time
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizerFast
import evaluate
from torch.utils.data import DataLoader
import os
import wget
import json
import torch_tensorrt
import modelopt.torch.quantization as mtq

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
metric = evaluate.load("squad")

# DataLoader
def collate(batch):
    questions = [x["question"] for x in batch]
    contexts = [x["context"] for x in batch]
    ids = [x["id"] for x in batch]
    answers = [x["answers"] for x in batch]

    tokens = tokenizer(
        questions,
        contexts,
        padding='max_length',
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

# Run model forward for calibration
def forward_loop(model):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"]
            )
            if i >= 15:
                break

# Extract static input shapes
for batch in dataloader:
    sample_input_ids = batch["input_ids"]
    sample_attention_mask = batch["attention_mask"]
    sample_token_type_ids = batch["token_type_ids"]
    static_shape = sample_input_ids.shape  # e.g., [8, 384]
    break

# Quantize using SmoothQuant (INT8)
print("🔧 Quantizing model using MTQ SmoothQuant...")
config = mtq.INT8_SMOOTHQUANT_CFG
model = mtq.quantize(model, config, forward_loop)

# Define model wrapper for torch.compile
class ModelWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

model_wrapper = ModelWrapper(model).to(device).eval()

# Compile with Torch-TensorRT
def compile_trt_model_static(fp16_mode=True):
    try:
        print(f"🚀 Compiling model with static shape: {static_shape}")
        compiled_model = torch.compile(
            model_wrapper,
            backend="torch_tensorrt",
            options={
                "truncate_long_and_double": True,
                "enabled_precisions": {torch.float16 if fp16_mode else torch.float32},
                "inputs": [
                    torch_tensorrt.Input(static_shape, dtype=torch.int32),  # input_ids
                    torch_tensorrt.Input(static_shape, dtype=torch.int32),  # attention_mask
                    torch_tensorrt.Input(static_shape, dtype=torch.int32),  # token_type_ids
                ]
            }
        )
        print("✅ Compiled model with Torch-TensorRT.")
        return compiled_model
    except Exception as e:
        print(f"⚠️ Error during compilation: {e}")
        print("➡️ Returning original wrapper.")
        return model_wrapper

trt_model = compile_trt_model_static(fp16_mode=True)

# Inference + Evaluation
latencies_trt = []
all_preds_trt, all_refs_trt = [], []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        if input_ids.shape != static_shape:
            raise ValueError(f"Input shape {input_ids.shape} does not match compiled shape {static_shape}")

        torch.cuda.synchronize()
        start = time.time()
        outputs = trt_model(input_ids, attention_mask, token_type_ids)
        torch.cuda.synchronize()
        end = time.time()

        latencies_trt.append(end - start)

        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()

        for i in range(len(input_ids)):
            start_idx = np.argmax(start_logits[i])
            end_idx = np.argmax(end_logits[i])
            input_tokens = input_ids[i].cpu().tolist()

            if start_idx > end_idx:
                pred_text = ""
            else:
                pred_text = tokenizer.decode(input_tokens[start_idx:end_idx+1], skip_special_tokens=True)

            all_preds_trt.append({'id': batch["id"][i], 'prediction_text': pred_text})
            all_refs_trt.append({'id': batch["id"][i], 'answers': batch["answers"][i]})

# Print sample predictions
print("\n==== TensorRT Sample Predictions ====")
for i in range(5):
    print(f"\nQ{i+1}: {all_refs_trt[i]['id']}")
    print(f"Question: {dataset[i]['question']}")
    print(f"Context: {dataset[i]['context'][:100]}...")
    print(f"Predicted: {all_preds_trt[i]['prediction_text']}")
    print(f"Actual: {all_refs_trt[i]['answers'][0]['text']}")

# Evaluation
results = metric.compute(predictions=all_preds_trt, references=all_refs_trt)
print(f"\n==== Torch-TensorRT Results ====")
print(f"Avg Latency per batch (batch_size=8): {np.mean(latencies_trt):.4f} sec")
print(f"F1 Score: {results['f1']:.2f}")
print(f"Exact Match (EM): {results['exact_match']:.2f}")
