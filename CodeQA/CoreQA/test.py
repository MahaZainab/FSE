#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless clusters
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==============================
# DEFAULTS (no args required)
# ==============================
DEFAULT_INPUT_JSON = "test.json"
DEFAULT_OUTPUT_JSON = "CodeLlama_CodeQA_student_test.json"
DEFAULT_OUTPUT_CSV = "CodeLlama_CodeQA_student_test.csv"
DEFAULT_PLOT_PNG = "CodeLlama_CodeQA_student_test.png"

# Model defaults: override by setting env var COREQA_MODEL if desired
DEFAULT_MODEL_NAME = os.environ.get("COREQA_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

# Generation defaults: override by env vars if desired
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("COREQA_MAX_NEW_TOKENS", "256"))
DEFAULT_TEMPERATURE = float(os.environ.get("COREQA_TEMPERATURE", "0.0"))

# Concurrency: GPU generation is not safely parallel by default. Keep 1 unless you know what you're doing.
MAX_WORKERS = int(os.environ.get("COREQA_MAX_WORKERS", "1"))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ==============================
# ORIGINAL PROMPTS (UNCHANGED)
# ==============================
SYSTEM_PROMPT = r"""
You are an expert system to assess the quality of code comprehension by an LLM. 
You will receive:
- A Python code snippet
- A programming question about that code
- A reference (correct) answer
- A model-generated answer (prediction)

Your task is to evaluate the prediction against the reference answer using four metrics:
accuracy, completeness, relevance, and clarity. For each, provide:
- An integer score from 1 to 3

### Accuracy
Compare the prediction with the reference to assess factual correctness and understanding of the code’s behavior and intent.
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
}
"""

EVAL_PROMPT_TEMPLATE = r"""
Code:
{code}

Question:
{question}

Reference Answer:
{reference}

Model Answer:
{prediction}
"""


# ==============================
# Utilities (from notebook)
# ==============================
def save_json_data_append(path: str, new_data: List[Dict[str, Any]]) -> None:
    """
    Load existing data from path, append new_data (list of dict),
    then save combined data back to path.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Combine and deduplicate by "id" if present, else just append
    existing_ids = {item.get("id") for item in existing_data if isinstance(item, dict) and "id" in item}
    filtered_new = [item for item in new_data if item.get("id") not in existing_ids]
    combined = existing_data + filtered_new

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)


def extract_all_scores(response_text: str) -> Dict[str, Dict[str, int]]:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return {}

    results = {}
    for metric, details in parsed.items():
        if isinstance(details, dict):
            score = details.get("score")
            if isinstance(score, int):
                results[metric] = {"score": score}
    return results


def export_csv(flat_records: List[Dict[str, Any]], path: str) -> None:
    df = pd.DataFrame(flat_records)
    df.to_csv(path, index=False)
    print(f"\nCSV exported to: {path}")


def visualize_scores(flat_records: List[Dict[str, Any]], out_png: str = DEFAULT_PLOT_PNG) -> None:
    df = pd.DataFrame(flat_records)
    if 'score' not in df.columns or 'metric' not in df.columns:
        raise ValueError("Expected 'score' and 'metric' columns in flat_records.")

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
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Score plot saved to: {out_png}")


def make_llm_eval_prompt(code: str, question: str, reference: str, prediction: str) -> str:
    # Uses the notebook's template verbatim
    return EVAL_PROMPT_TEMPLATE.format(
        code=code,
        question=question,
        reference=reference,
        prediction=prediction
    )


# ==============================
# HF Chat wrapper (based on singleturn_toms.py style)
# ==============================
class HFChat:
    def __init__(self, model_name: str, temperature: float = 0.0, max_new_tokens: int = 256):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()

    def invoke(self, chat_messages: List[Dict[str, str]]) -> str:
        # Prefer chat template if available (most instruct models provide it)
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: naive concatenation
            text = "\n\n".join([f"{m['role']}: {m['content']}" for m in chat_messages]) + "\nassistant:"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": float(self.temperature),
        }

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        gen_tokens = out[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# Create a single global model instance (important for GPU memory)
HF_LLM = HFChat(
    model_name=DEFAULT_MODEL_NAME,
    temperature=DEFAULT_TEMPERATURE,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS
)


async def call_llm_hf_async(prompt: str) -> str:
    """
    Async wrapper that matches the notebook's call style.
    Runs the blocking GPU generation in a thread (max_workers defaults to 1).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: HF_LLM.invoke(messages))


async def evaluate_batch_async(data: List[Dict[str, Any]], save_every: int = 25, output_path: str = DEFAULT_OUTPUT_JSON):
    results = []
    flat_records = []

    for i, item in enumerate(tqdm_asyncio(data, desc="Evaluating")):
        code = item["code"]
        question = item["question"]
        reference = item["answer"]
        prediction = item["prediction"]
        q_id = item.get("id", f"q{i+1}")

        prompt = make_llm_eval_prompt(code, question, reference, prediction)
        try:
            response_text = await call_llm_hf_async(prompt)
            metric_results = extract_all_scores(response_text)
        except Exception as e:
            metric_results = {}
            print(f"Error for id={q_id}: {e}")

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
                results = []  # clear buffer after saving
            except Exception as e:
                print(f"Warning: Failed to save at entry {i + 1}: {str(e)}")

    return results, flat_records


def main():
    input_path = DEFAULT_INPUT_JSON
    output_json_path = DEFAULT_OUTPUT_JSON
    output_csv_path = DEFAULT_OUTPUT_CSV

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file '{input_path}' not found. "
            f"Place it in the working directory or edit DEFAULT_INPUT_JSON in this script."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results, flat_records = asyncio.run(
        evaluate_batch_async(dataset, save_every=25, output_path=output_json_path)
    )

    # Save any leftover results after loop
    if results:
        save_json_data_append(output_json_path, results)

    export_csv(flat_records, path=output_csv_path)
    visualize_scores(flat_records)


if __name__ == "__main__":
    main()
