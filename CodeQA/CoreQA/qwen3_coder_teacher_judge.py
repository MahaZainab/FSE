#!/usr/bin/env python3
"""
qwen3_coder_teacher_judge.py

Run the same "LLM-as-a-judge" evaluation pipeline, but using a Hugging Face
Transformers model for inference.

This version is configured for:
  Qwen/Qwen3-Coder-30B-A3B-Instruct

It reads an input JSON list with items like:
  {
    "id": "...",              # optional
    "code": "...",
    "question": "...",
    "answer": "...",          # reference answer
    "prediction": "..."       # model answer being judged
  }

Outputs:
  1) <out>.json : list of records with wide numeric metric columns
  2) <out>.csv  : table with columns:
        code, question, answer, prediction, accuracy, completeness, relevance, clarity
     (plus id if present)

Notes:
- Requires: torch, transformers, accelerate, tqdm, pandas
- Run (GPU recommended):
    python qwen3_coder_teacher_judge.py --input mini.json --out qwen3_coder_mini
"""

import argparse
import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Concurrency (same pattern)
# -----------------------------
EXECUTOR = ThreadPoolExecutor(max_workers=5)


# -----------------------------
# HF model config
# -----------------------------
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
_tokenizer = None
_model = None


def get_hf_model(model_name: str):
    """Lazy-load tokenizer/model once per process."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
            _tokenizer.pad_token = _tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        # Safer on a variety of clusters/versions
        try:
            _model.config.use_cache = False
        except Exception:
            pass
        try:
            _model.generation_config.use_cache = False
        except Exception:
            pass

        _model.eval()
    return _tokenizer, _model


class _HFResponse:
    def __init__(self, content: str):
        self.content = content


class HFChat:
    """Drop-in-ish adapter: response = llm.invoke(messages); return response.content"""

    def __init__(self, model_name: str, temperature: float = 0.0, max_new_tokens: int = 256):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def invoke(self, messages):
        tokenizer, model = get_hf_model(self.model_name)

        chat = []
        for m in messages:
            role = "system" if m.__class__.__name__ == "SystemMessage" else "user"
            chat.append({"role": role, "content": m.content})

        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature and self.temperature > 0),
            use_cache=False,
        )
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(self.temperature)

        if tokenizer.eos_token_id is not None:
            gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        # IMPORTANT: slice by token count (avoid fragile string slicing)
        gen_tokens = out[0, inputs["input_ids"].shape[-1] :]
        generated = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return _HFResponse(generated)


# -----------------------------
# Message stubs (no LangChain dep)
# -----------------------------
class SystemMessage:
    def __init__(self, content: str):
        self.content = content


class HumanMessage:
    def __init__(self, content: str):
        self.content = content


SYSTEM_PROMPT = """
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

Score meanings:
- 1: Completely incorrect or irrelevant; does not address the reference answer.
- 2: Partially correct; some key facts are accurate, but major details are wrong or missing.
- 3: Fully correct; matches the reference answer in meaning and factual content.

### Completeness
Check if the prediction covers all important parts of the reference answer.

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
Evaluate how clearly and logically the prediction is expressed.

Score meanings:
- 1: Confusing, vague, or incoherent.
- 2: Understandable but awkwardly phrased or slightly unclear.
- 3: Clear, concise, and easy to follow.

Final Instructions:
Base your evaluation strictly on the content provided. Do not hallucinate missing information.
Respond ONLY with a JSON object in this exact format:
{
  "accuracy": { "score": 1-3 },
  "completeness": { "score": 1-3 },
  "relevance": { "score": 1-3 },
  "clarity": { "score": 1-3 }
}
""".strip()


def make_llm_eval_prompt(code: str, question: str, reference: str, prediction: str) -> str:
    return f"""Code:
{code}

Question:
{question}

Reference Answer:
{reference}

Model Answer:
{prediction}
"""


def _extract_first_json_object(text: str) -> Optional[str]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else None


def extract_scores(response_text: str) -> Dict[str, int]:
    """
    Returns: {"accuracy": 1-3, "completeness": 1-3, "relevance": 1-3, "clarity": 1-3}
    Missing/invalid -> omitted.
    """
    blob = _extract_first_json_object(response_text)
    if not blob:
        return {}

    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return {}

    out: Dict[str, int] = {}
    for metric in ("accuracy", "completeness", "relevance", "clarity"):
        details = parsed.get(metric)
        if not isinstance(details, dict):
            continue
        score = details.get("score")

        # tolerate "3" or 2.0
        if isinstance(score, str) and score.strip().isdigit():
            score = int(score.strip())
        elif isinstance(score, float) and score.is_integer():
            score = int(score)

        if isinstance(score, int) and 1 <= score <= 3:
            out[metric] = score

    return out


async def call_judge_async(llm: HFChat, prompt: str) -> str:
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(EXECUTOR, lambda: llm.invoke(messages))
    return resp.content


async def evaluate_dataset(
    dataset: List[Dict[str, Any]],
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    save_every: int,
    out_json_path: str,
) -> List[Dict[str, Any]]:
    llm = HFChat(model_name=model_name, temperature=temperature, max_new_tokens=max_new_tokens)

    results_buffer: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    def _append_save(batch: List[Dict[str, Any]]):
        if not batch:
            return
        try:
            with open(out_json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []
        existing.extend(batch)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    for i, item in enumerate(tqdm_asyncio(dataset, desc="Evaluating")):
        q_id = item.get("id", f"q{i+1}")
        code = item.get("code", "")
        question = item.get("question", "")
        reference = item.get("answer", item.get("reference", ""))
        prediction = item.get("prediction", "")

        prompt = make_llm_eval_prompt(code, question, reference, prediction)

        try:
            response = await call_judge_async(llm, prompt)
            scores = extract_scores(response)
        except Exception as e:
            print(f"[WARN] error at {q_id}: {e}")
            scores = {}

        row = {
            "id": q_id,
            "code": code,
            "question": question,
            "answer": reference,
            "prediction": prediction,
            "accuracy": scores.get("accuracy"),
            "completeness": scores.get("completeness"),
            "relevance": scores.get("relevance"),
            "clarity": scores.get("clarity"),
        }

        results_buffer.append(row)
        all_results.append(row)

        if save_every > 0 and (i + 1) % save_every == 0:
            _append_save(results_buffer)
            results_buffer = []

    # final flush
    _append_save(results_buffer)
    return all_results


def export_csv_wide(rows: List[Dict[str, Any]], out_csv_path: str):
    # Match the spreadsheet format in your screenshot (keep id as first column if present)
    df = pd.DataFrame(rows)
    # Ensure column order
    cols = [c for c in ["id", "code", "question", "answer", "prediction", "accuracy", "completeness", "relevance", "clarity"] if c in df.columns]
    df = df[cols]
    df.to_csv(out_csv_path, index=False)
    print(f"Saved CSV -> {out_csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON file (list of examples)")
    ap.add_argument("--out", required=True, help="Output prefix (writes <out>.json and <out>.csv)")
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model name")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--save_every", type=int, default=25)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Input JSON must be a list of examples")

    out_json = f"{args.out}.json"
    out_csv = f"{args.out}.csv"

    rows = asyncio.run(
        evaluate_dataset(
            dataset=dataset,
            model_name=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            save_every=args.save_every,
            out_json_path=out_json,
        )
    )
    export_csv_wide(rows, out_csv)


if __name__ == "__main__":
    main()
