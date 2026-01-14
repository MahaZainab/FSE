import json
import random
import re
from typing import Any, Dict, Optional, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================
# Experiment: Single teacher intervention (Theory-of-Mind guidance) -> student updates scores once
# - Teacher intervention is ONE message (no teacher-student dialogue, no multi-turn mentoring loop)
# - Teacher applies Theory of Mind to diagnose the student's evaluation and give targeted guidance
# - Student re-scores once using the teacher's ToM guidance
# =============================

INPUT_JSON = "mini.json"                         # expects list of dicts with keys: code, question, answer, prediction
OUTPUT_JSON = "rq3_single_tom_then_rescore_mini.json"
OUTPUT_CSV = "rq3_single_tom_then_rescore_mini.csv"

STUDENT_JUDGE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TEACHER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

TEACHER_INTERVENTION_RATE = 0.30
RANDOM_SEED = 42


# -----------------------------
# Model wrapper (HF)
# -----------------------------
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
        prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": float(self.temperature),
        }

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        gen_tokens = out[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# -----------------------------
# Anchored rubric (explicit meaning of 1/2/3)
# -----------------------------
ANCHORED_RUBRIC = r"""
Scoring uses an ANCHORED 3-point rubric:

GLOBAL ANCHORS:
- Score 3: Essentially correct and expert-level for this task.
  * Matches the reference answer in meaning (semantic equivalence).
  * Correctly reflects the code’s behavior and intent.
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


# -----------------------------
# Student prompt (judge): scores only (strict JSON)
# -----------------------------
JUDGE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing the performance of a Teaching Assistant (TA)
in an introductory Python programming course.

You will receive:
- A Python code snippet
- A student question about that code
- A reference (correct) answer
- A TA LLM-generated answer (called the prediction)

Evaluate the TA prediction on four dimensions (Accuracy, Completeness, Relevance, Clarity).

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


def build_judge_user_prompt(code: str, question: str, reference: str, prediction: str) -> str:
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


# -----------------------------
# Teacher prompt: SINGLE intervention using Theory of Mind 
# -----------------------------
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


def build_teacher_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    student_scores: Dict[str, int],
) -> str:
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


# -----------------------------
# Student rescore prompt (after teacher ToM guidance): strict JSON only
# -----------------------------
RESCORE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing a TA answer.

You will receive:
- code, question, reference answer, TA prediction
- your previous scores
- teacher Theory-of-Mind guidance (including teacher_scores and checklist)

Use the SAME anchored rubric:

{ANCHORED_RUBRIC}

Update your scores accordingly.

Return ONLY valid JSON in EXACTLY this format:
{{
  "accuracy": {{ "score": 1-3 }},
  "completeness": {{ "score": 1-3 }},
  "relevance": {{ "score": 1-3 }},
  "clarity": {{ "score": 1-3 }}
}}

Do NOT include explanations.
""".strip()


def build_rescore_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    prev_scores: Dict[str, int],
    teacher_json: Dict[str, Any],
) -> str:
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


# -----------------------------
# Parsing helpers
# -----------------------------
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _extract_first_json_obj(text: str) -> Optional[str]:
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    js_text = _extract_first_json_obj(text)
    if not js_text:
        return None
    try:
        obj = json.loads(js_text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_score_json(raw_text: str) -> Optional[Dict[str, int]]:
    obj = _safe_json(raw_text)
    if not obj:
        return None
    out: Dict[str, int] = {}
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = obj.get(k, {})
        score = v.get("score") if isinstance(v, dict) else None
        if isinstance(score, str) and score.isdigit():
            score = int(score)
        if isinstance(score, float):
            score = int(round(score))
        if not isinstance(score, int) or score < 1 or score > 3:
            return None
        out[k] = score
    return out


def parse_teacher_json(raw_text: str) -> Optional[Dict[str, Any]]:
    obj = _safe_json(raw_text)
    if not obj:
        return None
    required = {"teacher_scores", "score_comparison", "inferred_student_mental_model", "tom_guidance", "checklist_next_time"}
    if not required.issubset(set(obj.keys())):
        return None
    ts = obj.get("teacher_scores", {})
    if not isinstance(ts, dict):
        return None
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = ts.get(k)
        if not isinstance(v, int) or v < 1 or v > 3:
            return None
    return obj


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    random.seed(RANDOM_SEED)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    student = HFChat(STUDENT_JUDGE_MODEL, temperature=0.0, max_new_tokens=256)
    teacher = HFChat(TEACHER_MODEL, temperature=0.0, max_new_tokens=900)

    n = len(data)
    k = int(round(TEACHER_INTERVENTION_RATE * n))
    intervene_ids = set(random.sample(range(n), k)) if n > 0 and k > 0 else set()

    records = []

    for idx, ex in enumerate(data):
        code = ex.get("code", "")
        question = ex.get("question", "")
        reference = ex.get("answer", "")
        prediction = ex.get("prediction", "")

        # Student initial scoring
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

        if teacher_intervened and student_scores is not None:
            # Teacher single ToM intervention
            raw_teacher = teacher.invoke([
                {"role": "system", "content": TEACHER_TOM_SYSTEM_PROMPT},
                {"role": "user", "content": build_teacher_user_prompt(code, question, reference, prediction, student_scores)},
            ])
            teacher_out = parse_teacher_json(raw_teacher)

            # Student updates scores once based on teacher guidance
            if teacher_out is not None:
                raw_rescore = student.invoke([
                    {"role": "system", "content": RESCORE_SYSTEM_PROMPT},
                    {"role": "user", "content": build_rescore_user_prompt(code, question, reference, prediction, student_scores, teacher_out)},
                ])
                rescored_scores = parse_score_json(raw_rescore)

        final_scores = rescored_scores if rescored_scores is not None else (student_scores or {})

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

            # flattened final
            "accuracy": final_scores.get("accuracy"),
            "completeness": final_scores.get("completeness"),
            "relevance": final_scores.get("relevance"),
            "clarity": final_scores.get("clarity"),

            # initial vs rescored
            "accuracy_initial": student_scores.get("accuracy") if student_scores else None,
            "completeness_initial": student_scores.get("completeness") if student_scores else None,
            "relevance_initial": student_scores.get("relevance") if student_scores else None,
            "clarity_initial": student_scores.get("clarity") if student_scores else None,

            "accuracy_rescored": rescored_scores.get("accuracy") if rescored_scores else None,
            "completeness_rescored": rescored_scores.get("completeness") if rescored_scores else None,
            "relevance_rescored": rescored_scores.get("relevance") if rescored_scores else None,
            "clarity_rescored": rescored_scores.get("clarity") if rescored_scores else None,
        }
        records.append(record)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(records)
    df_out = df[[
        "code", "question", "answer", "prediction",
        "accuracy", "completeness", "relevance", "clarity",
        "accuracy_initial", "completeness_initial", "relevance_initial", "clarity_initial",
        "accuracy_rescored", "completeness_rescored", "relevance_rescored", "clarity_rescored",
        "teacher_intervened"
    ]]
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("Done")
    print("JSON:", OUTPUT_JSON)
    print("CSV :", OUTPUT_CSV)
    print(f"Teacher intervened on {len(intervene_ids)}/{n} examples ({TEACHER_INTERVENTION_RATE*100:.0f}%).")


if __name__ == "__main__":
    main()
