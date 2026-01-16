#!/usr/bin/env python3
"""
Single-turn Theory-of-Mind Supervision (ToMS) Evaluation
Adapted for remote cluster execution with GPU support

Experiment: Single teacher intervention (Theory-of-Mind guidance) -> student updates scores once
- Teacher intervention is ONE message (no teacher-student dialogue, no multi-turn mentoring loop)
- Teacher applies Theory of Mind to diagnose the student's evaluation and give targeted guidance
- Student re-scores once using the teacher's ToM guidance
"""

import json
import random
import re
import os
from typing import Any, Dict, Optional, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =============================
# Configuration
# =============================
INPUT_JSON = "mini.json"  # expects: code, question, answer, prediction
OUTPUT_JSON = "gptprompt.json"
OUTPUT_CSV = "gptprompt.csv"

STUDENT_JUDGE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TEACHER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

TEACHER_INTERVENTION_RATE = 0.30
RANDOM_SEED = 42

# HuggingFace cache directory
HF_CACHE_DIR = os.getenv("HF_HOME", "/aiau010_scratch/maz0032/.cache/huggingface")

# =============================
# Model wrapper (HF)
# =============================
class HFChat:
    """HuggingFace chat model wrapper with GPU support."""
    
    def __init__(
        self, 
        model_name: str, 
        temperature: float = 0.0, 
        max_new_tokens: int = 256,
        cache_dir: str = HF_CACHE_DIR
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        print(f"\nLoading model: {model_name}")
        print(f"Cache directory: {cache_dir}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded on device: {self.model.device}")

    def invoke(self, chat_messages: List[Dict[str, str]]) -> str:
        """Generate response from chat messages."""
        prompt = self.tokenizer.apply_chat_template(
            chat_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": float(self.temperature) if self.temperature > 0 else 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        gen_tokens = out[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# =============================
# Prompts and Rubric
# =============================
ANCHORED_RUBRIC = r"""
Scoring uses an ANCHORED 3-point rubric:

GLOBAL ANCHORS:
- Score 3: Essentially correct and expert-level for this task.
  * Matches the reference answer in meaning (semantic equivalence).
  * Correctly reflects the code's behavior and intent.
  * Contains no major errors or misleading statements.

- Score 2: Partially correct but flawed.
  * Some core facts are correct.
  * However, important details/conditions are missing, incorrect, ambiguous, or misweighted.
  * A student relying on it could be misled or left with an incomplete understanding.

- Score 1: Fails to answer the question.
  * Mostly incorrect, irrelevant, or contradicts the code/reference.
  * Does not demonstrate real understanding of the problem.

DIMENSIONS:
Accuracy: correctness vs reference/code behavior.
Completeness: coverage of all essential points from the reference.
Relevance: directly answers the question, stays on-topic.
Clarity: easy to follow, unambiguous, well-structured.
""".strip()

JUDGE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing the quality of code comprehension by an LLM

You will receive:
- A Python code snippet
- A student question about that code
- A reference (correct) answer
- A TA LLM-generated answer (called the prediction)

Your task is to evaluate the prediction against the reference answer using four metrics:
accuracy, completeness, relevance, and clarity. For each, provide:
{ANCHORED_RUBRIC}

Return ONLY valid JSON in EXACTLY this format:
{{
  "accuracy": {{ "score": 1-3 }},
  "completeness": {{ "score": 1-3 }},
  "relevance": {{ "score": 1-3 }},
  "clarity": {{ "score": 1-3 }}
}}

Do NOT include explanations.
""".strip()

TEACHER_TOM_SYSTEM_PROMPT = f"""
You are a 30B teacher LLM supervising a student LLM-as-judge.

You will receive:
- code, question, reference answer, TA prediction
- student judge scores (1-3) for accuracy/completeness/relevance/clarity

Use the SAME anchored rubric:

{ANCHORED_RUBRIC}

Your job is a SINGLE intervention (one response) that uses THEORY OF MIND:
1) Independently form your own scores based on code/question/reference/prediction.
2) Compare your scores to the student's scores.
3) Infer the student's likely mental model from the mismatch patterns (e.g., what it over-weighted, overlooked, assumed).
4) Provide targeted guidance that updates the student's evaluation strategy.

- Keep guidance concrete and evidence-based (point to specific missing/wrong aspects in the TA prediction vs reference/code).

Return ONLY valid JSON in EXACTLY this format:

{{
  "teacher_scores": {{
    "accuracy": 1-3,
    "completeness": 1-3,
    "relevance": 1-3,
    "clarity": 1-3
  }},
  "score_comparison": {{
    "accuracy": {{"student": 1-3, "teacher": 1-3, "match": true/false}},
    "completeness": {{"student": 1-3, "teacher": 1-3, "match": true/false}},
    "relevance": {{"student": 1-3, "teacher": 1-3, "match": true/false}},
    "clarity": {{"student": 1-3, "teacher": 1-3, "match": true/false}}
  }},
  "inferred_student_mental_model": {{
    "beliefs_or_criteria": "what the student judge seems to optimize for",
    "likely_blind_spots": ["...","..."],
    "likely_assumptions": ["...","..."]
  }},
  "tom_guidance": {{
    "accuracy": "2-4 sentences if mismatch else empty string",
    "completeness": "2-4 sentences if mismatch else empty string",
    "relevance": "2-4 sentences if mismatch else empty string",
    "clarity": "2-4 sentences if mismatch else empty string"
  }},
  "checklist_next_time": ["short actionable item", "short actionable item"]
}}

Rules:
- In tom_guidance, leave empty string for dimensions where match=true.
- Keep the mental model concise and plausible based on the student's scores.
""".strip()

RESCORE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing a the code comprehension by an LLM.

You will receive:
- code, question, reference answer, TA prediction
- your previous scores
- You will now see Theory-of-Mind guidance from a teacher describing:
- how you may have been thinking,
- what conceptual gaps or misconceptions you likely had,
- and how to improve.
IMPORTANT:
• Do not use teacher scores.
• You must NOT try to infer or guess any teacher scores.
• You must rescore ONLY by re-evaluating the work itself using the rubric.

Your job is to reflect, correct your understanding, and then rescore.
Use the SAME anchored rubric:
{ANCHORED_RUBRIC}

Return ONLY valid JSON in EXACTLY this format:
{{
  "accuracy": {{ "score": 1-3 }},
  "completeness": {{ "score": 1-3 }},
  "relevance": {{ "score": 1-3 }},
  "clarity": {{ "score": 1-3 }}
}}

Do NOT include explanations.
""".strip()


# =============================
# Prompt builders
# =============================
def build_judge_user_prompt(code: str, question: str, reference: str, prediction: str) -> str:
    """Build initial judge prompt."""
    return f"""
Code:
```python
{code}
```

Question:
{question}

Reference Answer:
{reference}

Prediction:
{prediction}
""".strip()


def build_teacher_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    student_scores: Dict[str, int],
) -> str:
    """Build teacher intervention prompt."""
    return f"""
Code:
```python
{code}
```

Question:
{question}

Reference Answer:
{reference}

TA Prediction:
{prediction}

Student Judge Scores:
{json.dumps(student_scores, indent=2)}
""".strip()


def build_rescore_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    prev_scores: Dict[str, int],
    teacher_json: Dict[str, Any],
) -> str:
    """Build student rescore prompt after teacher guidance."""
    return f"""
Code:
```python
{code}
```

Question:
{question}

Reference Answer:
{reference}

Prediction:
{prediction}

Your Previous Scores:
{json.dumps(prev_scores, indent=2)}

Teacher Guidance JSON:
{json.dumps(teacher_json, indent=2, ensure_ascii=False)}

Now update your scores and return ONLY the strict score JSON.
""".strip()


# =============================
# Parsing helpers
# =============================
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _extract_first_json_obj(text: str) -> Optional[str]:
    """Extract first JSON object from text."""
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON from text."""
    js_text = _extract_first_json_obj(text)
    if not js_text:
        return None
    try:
        obj = json.loads(js_text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_score_json(raw_text: str) -> Optional[Dict[str, int]]:
    """Parse score JSON with validation."""
    obj = _safe_json(raw_text)
    if not obj:
        return None
    
    out: Dict[str, int] = {}
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = obj.get(k, {})
        score = v.get("score") if isinstance(v, dict) else None
        
        # Handle string numbers
        if isinstance(score, str) and score.isdigit():
            score = int(score)
        # Handle floats
        if isinstance(score, float):
            score = int(round(score))
        
        # Validate range
        if not isinstance(score, int) or score < 1 or score > 3:
            return None
        
        out[k] = score
    
    return out


def parse_teacher_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse teacher JSON with validation."""
    obj = _safe_json(raw_text)
    if not obj:
        return None
    
    required = {
        "teacher_scores", 
        "score_comparison", 
        "inferred_student_mental_model", 
        "tom_guidance", 
        "checklist_next_time"
    }
    
    if not required.issubset(set(obj.keys())):
        return None
    
    # Validate teacher scores
    ts = obj.get("teacher_scores", {})
    if not isinstance(ts, dict):
        return None
    
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = ts.get(k)
        if not isinstance(v, int) or v < 1 or v > 3:
            return None
    
    return obj


# =============================
# Main execution
# =============================
def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("Single-Turn Theory-of-Mind Supervision (ToMS)")
    print("=" * 60)
    
    # Set random seed
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Load data
    print(f"\nLoading data from: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    # Initialize models
    print("\n" + "=" * 60)
    print("Initializing models...")
    print("=" * 60)
    
    student = HFChat(
        STUDENT_JUDGE_MODEL, 
        temperature=0.0, 
        max_new_tokens=256,
        cache_dir=HF_CACHE_DIR
    )
    
    teacher = HFChat(
        TEACHER_MODEL, 
        temperature=0.0, 
        max_new_tokens=900,
        cache_dir=HF_CACHE_DIR
    )

    # Determine intervention examples
    n = len(data)
    k = int(round(TEACHER_INTERVENTION_RATE * n))
    intervene_ids = set(random.sample(range(n), k)) if n > 0 and k > 0 else set()
    
    print(f"\n" + "=" * 60)
    print(f"Will intervene on {k}/{n} examples ({TEACHER_INTERVENTION_RATE*100:.0f}%)")
    print("=" * 60 + "\n")

    records = []

    # Process each example
    for idx, ex in enumerate(tqdm(data, desc="Processing examples")):
        code = ex.get("code", "")
        question = ex.get("question", "")
        reference = ex.get("answer", "")
        prediction = ex.get("prediction", "")

        # Step 1: Student initial scoring
        raw_student = student.invoke([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(code, question, reference, prediction)},
        ])
        student_scores = parse_score_json(raw_student)

        teacher_intervened = idx in intervene_ids
        raw_teacher = None
        teacher_out = None
        raw_rescore = None
        rescored_scores = None

        # Step 2: Teacher intervention (if selected)
        if teacher_intervened and student_scores is not None:
            raw_teacher = teacher.invoke([
                {"role": "system", "content": TEACHER_TOM_SYSTEM_PROMPT},
                {"role": "user", "content": build_teacher_user_prompt(
                    code, question, reference, prediction, student_scores
                )},
            ])
            teacher_out = parse_teacher_json(raw_teacher)

            # Step 3: Student rescores based on teacher guidance
            if teacher_out is not None:
                raw_rescore = student.invoke([
                    {"role": "system", "content": RESCORE_SYSTEM_PROMPT},
                    {"role": "user", "content": build_rescore_user_prompt(
                        code, question, reference, prediction, student_scores, teacher_out
                    )},
                ])
                rescored_scores = parse_score_json(raw_rescore)

        # Determine final scores
        final_scores = rescored_scores if rescored_scores is not None else (student_scores or {})

        # Build record
        record = {
            "id": ex.get("id", idx),
            "code": code,
            "question": question,
            "answer": reference,
            "prediction": prediction,

            "raw_student_judge_output": raw_student,
            "initial_student_scores": student_scores,

            "teacher_intervened": teacher_intervened,
            "raw_teacher_output": raw_teacher,
            "teacher_tom_json": teacher_out,

            "raw_student_rescore_output": raw_rescore,
            "rescored_student_scores": rescored_scores,

            # Final scores (flattened)
            "accuracy": final_scores.get("accuracy"),
            "completeness": final_scores.get("completeness"),
            "relevance": final_scores.get("relevance"),
            "clarity": final_scores.get("clarity"),

            # Initial scores
            "accuracy_initial": student_scores.get("accuracy") if student_scores else None,
            "completeness_initial": student_scores.get("completeness") if student_scores else None,
            "relevance_initial": student_scores.get("relevance") if student_scores else None,
            "clarity_initial": student_scores.get("clarity") if student_scores else None,

            # Rescored scores
            "accuracy_rescored": rescored_scores.get("accuracy") if rescored_scores else None,
            "completeness_rescored": rescored_scores.get("completeness") if rescored_scores else None,
            "relevance_rescored": rescored_scores.get("relevance") if rescored_scores else None,
            "clarity_rescored": rescored_scores.get("clarity") if rescored_scores else None,
        }
        records.append(record)

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {OUTPUT_JSON}")

    # Save CSV
    df = pd.DataFrame(records)
    df_out = df[[
        "id", "code", "question", "answer", "prediction",
        "accuracy", "completeness", "relevance", "clarity",
        "accuracy_initial", "completeness_initial", "relevance_initial", "clarity_initial",
        "accuracy_rescored", "completeness_rescored", "relevance_rescored", "clarity_rescored",
        "teacher_intervened"
    ]]
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"CSV saved: {OUTPUT_CSV}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total examples: {n}")
    print(f"Teacher interventions: {len(intervene_ids)} ({TEACHER_INTERVENTION_RATE*100:.0f}%)")
    print(f"Output files:")
    print(f"  - {OUTPUT_JSON}")
    print(f"  - {OUTPUT_CSV}")
    print("\nDone!")


if __name__ == "__main__":
    main()