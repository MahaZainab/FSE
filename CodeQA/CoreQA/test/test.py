#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation Script for HuggingFace Models
Adapted for remote cluster execution with GPU support
"""

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # or "meta-llama/Llama-2-7b-chat-hf"
INPUT_FILE = "test.json"
OUTPUT_JSON = "CodeQA_test.json"
OUTPUT_CSV = "CodeQA_test.csv"
OUTPUT_PLOT = "evaluation_scores.png"
SAVE_EVERY = 25
MAX_WORKERS = 4
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0

# Initialize model and tokenizer globally
print(f"Loading model: {MODEL_NAME}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=os.getenv("HF_HOME", "/aiau010_scratch/maz0032/.cache/huggingface"),
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=os.getenv("HF_HOME", "/aiau010_scratch/maz0032/.cache/huggingface"),
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded on device: {model.device}")

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


SYSTEM_PROMPT = """You are an expert system to assess the quality of code comprehension by an LLM. 
You will receive:
- A Python code snippet
- A programming question about that code
- A reference (correct) answer
- A model-generated answer (prediction)

Your task is to evaluate the prediction against the reference answer using four metrics:
accuracy, completeness, relevance, and clarity. For each, provide:
- An integer score from 1 to 3

### Accuracy
Compare the prediction with the reference to assess factual correctness and understanding of the code's behavior and intent.
You must judge whether the prediction reflects accurate behavior and matches core facts from the reference. 
You need to consider semantic meaning of code comprehension: understanding the structure, functionality, and intent behind the code.

Score meanings:
- 1: Completely incorrect or irrelevant; does not address the reference answer.
- 2: Partially correct; some key facts are accurate, but major details are wrong or missing.
- 3: Fully correct; matches the reference answer in meaning and factual content.

### Completeness
Check if the prediction covers all important parts of the reference answer, including key concepts or conditions.

Score meanings:
- 1: Omits most key information or contains only a tiny fragment of relevant content.
- 2: Covers some elements but misses important parts.
- 3: Fully covers all essential information from the reference.

### Relevance
Assess whether the prediction directly addresses the question and stays on-topic.

Score meanings:
- 1: Completely irrelevant or mostly unrelated.
- 2: Partially related but misses the main point.
- 3: Fully focused and directly answers the question.

### Clarity
Evaluate how clearly and logically the prediction is expressed, ensuring it is easy to understand.

Score meanings:
- 1: Confusing, vague, or incoherent.
- 2: Understandable but awkwardly phrased or slightly unclear.
- 3: Clear, concise, and easy to follow.

Example:

{
  "Code": "def aggregate metadata get by host context host key None return IMPL aggregate metadata get by host context host key",
  "Question": "What does the code get ?",
  "Reference Answer": "metadata for all aggregates that host belongs to",
  "Model Answer": "Hosts",
  "Evaluation": {
    "accuracy": { "score": 1 },
    "completeness": { "score": 1 },
    "relevance": { "score": 2 },
    "clarity": { "score": 3 }
  }
}

Final Instructions:
Base your evaluation strictly on the content provided. Do not hallucinate missing information. Be consistent and objective.
Provide evaluation only in integer score only for each dimension.
Respond only with a JSON object in this exact format:
{
"accuracy": { "score": 1-3 },
"completeness": { "score": 1-3 },
"relevance": { "score": 1-3 },
"clarity": { "score": 1-3 }
}"""


def call_llm_hf(prompt, temperature=TEMPERATURE):
    """Call HuggingFace model with the given prompt."""
    # Format for Llama-3 chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response


def save_json_data_append(path, new_data):
    """Load existing data, append new_data, save combined data."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_ids = {item.get("id") for item in existing_data if "id" in item}
    filtered_new = [item for item in new_data if item.get("id") not in existing_ids]
    combined = existing_data + filtered_new

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)


def make_llm_eval_prompt(code, question, reference, prediction):
    """Create evaluation prompt."""
    return f"""Code:
{code}

Question:
{question}

Reference Answer:
{reference}

Model Answer:
{prediction}"""


def extract_all_scores(response_text):
    """Extract scores from JSON response."""
    try:
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            parsed = json.loads(json_str)
        else:
            parsed = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Failed to parse response: {response_text[:200]}")
        return {}

    results = {}
    for metric, details in parsed.items():
        if isinstance(details, dict):
            score = details.get("score")
            if isinstance(score, int):
                results[metric] = {"score": score}
    return results


def evaluate_batch(data, save_every=SAVE_EVERY, output_path=OUTPUT_JSON):
    """Evaluate batch of data."""
    results = []
    flat_records = []

    for i, item in enumerate(tqdm(data, desc="Evaluating")):
        code = item["code"]
        question = item["question"]
        reference = item["answer"]
        prediction = item["prediction"]
        q_id = item.get("id", f"q{i+1}")

        prompt = make_llm_eval_prompt(code, question, reference, prediction)
        try:
            response = call_llm_hf(prompt)
            metric_results = extract_all_scores(response)
        except Exception as e:
            print(f"Error evaluating entry {i+1}: {e}")
            metric_results = {}

        sample_result = {
            "id": q_id,
            "code": code,
            "question": question,
            "reference": reference,
            "prediction": prediction,
        }

        for metric in ["accuracy", "completeness", "relevance", "clarity"]:
            score = metric_results.get(metric, {}).get("score", None)
            sample_result[metric] = {"score": score}
            flat_records.append({
                "id": q_id,
                "code": code,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "metric": metric,
                "score": score,
            })

        results.append(sample_result)

        if (i + 1) % save_every == 0 or (i + 1) == len(data):
            try:
                save_json_data_append(output_path, results)
                print(f"Appended partial results after {i + 1} entries.")
                results = []
            except Exception as e:
                print(f"Warning: Failed to save at entry {i + 1}: {str(e)}")

    return results, flat_records


def export_csv(flat_records, path):
    """Export results to CSV."""
    df = pd.DataFrame(flat_records)
    df.to_csv(path, index=False)
    print(f"\nCSV exported to: {path}")


def visualize_scores(flat_records, output_path=OUTPUT_PLOT):
    """Create visualization of scores."""
    df = pd.DataFrame(flat_records)
    if 'score' not in df.columns or 'metric' not in df.columns:
        raise ValueError("Expected 'score' and 'metric' columns in flat_records.")

    # Remove None values
    df = df[df['score'].notna()]
    
    grouped = df.groupby("metric")["score"]
    avg_scores = grouped.mean()
    std_scores = grouped.std().fillna(0)

    plt.figure(figsize=(8, 5))
    plt.bar(
        avg_scores.index,
        avg_scores.values,
        yerr=std_scores.values,
        capsize=5,
        color='skyblue',
        edgecolor='black'
    )
    plt.title("Average Scores per Metric")
    plt.ylabel("Score (1–3)")
    plt.ylim(1, 3)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def main():
    """Main execution function."""
    print("Starting evaluation...")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output JSON: {OUTPUT_JSON}")
    print(f"Output CSV: {OUTPUT_CSV}")
    
    # Load dataset
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Evaluate
    results, flat_records = evaluate_batch(
        dataset,
        save_every=SAVE_EVERY,
        output_path=OUTPUT_JSON
    )
    
    # Save any leftover results
    if results:
        save_json_data_append(OUTPUT_JSON, results)
    
    # Export CSV
    export_csv(flat_records, path=OUTPUT_CSV)
    
    # Create visualization
    visualize_scores(flat_records, output_path=OUTPUT_PLOT)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()