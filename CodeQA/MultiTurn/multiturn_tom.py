import json
import random
import re
from typing import Any, Dict, Optional, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================
# Configuration
# =============================
INPUT_JSON = "mini.json"                         # list of dicts with keys: code, question, answer, prediction
OUTPUT_JSON = "rq3_multiturn_mini.json"
OUTPUT_CSV = "rq3_multiturn_mini.csv"

STUDENT_JUDGE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TEACHER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Teacher intervention rate (e.g., 30% of examples)
TEACHER_INTERVENTION_RATE = 0.30

RANDOM_SEED = 42
# =============================


# -----------------------------
# Model wrapper (HF)
# -----------------------------
class HFChat:
    def __init__(self, model_name: str, temperature: float = 0.0, max_new_tokens: int = 512):
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
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return text


# -----------------------------
# Shared anchored rubric text (used everywhere)
# -----------------------------
ANCHORED_RUBRIC = r"""
Scoring uses an ANCHORED 3-point rubric. Use these anchors consistently across all dimensions:

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

DIMENSION-SPECIFIC ANCHORS:

Accuracy:
- 3: Semantically matches the reference and code behavior; no major factual errors.
- 2: Mix of correct and incorrect claims; major mistake(s) or missing/incorrect key behavior.
- 1: Mostly wrong or unrelated to code/reference.

Completeness:
- 3: Covers all essential points from the reference (including key conditions/cases).
- 2: Covers some essentials but misses important points/conditions.
- 1: Omits most essential information.

Relevance:
- 3: Directly answers the question; stays on-topic; no major distractions.
- 2: On-topic but misses the main point OR adds distracting/irrelevant content.
- 1: Largely off-topic or does not address the asked question.

Clarity:
- 3: Clear, well-structured, unambiguous; easy for a student to follow.
- 2: Understandable but awkward, vague, or poorly structured; potential ambiguity.
- 1: Confusing, incoherent, or hard to interpret.
""".strip()


# -----------------------------
# Prompts
# -----------------------------
# Turn 1: student outputs scores + reasoning + ideal answer belief (JSON)
STUDENT_EXPLAIN_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing the performance of a Teaching Assistant (TA)
in an introductory Python programming course.

You will receive:
- A Python code snippet
- A student question about that code
- A reference (correct) answer
- A TA LLM-generated answer (called the prediction)

Evaluate the TA prediction on four dimensions:
Accuracy, Completeness, Relevance, Clarity.

{ANCHORED_RUBRIC}

TASK:
1) Assign an integer score from 1 to 3 for each dimension using the anchored rubric.
2) Explain your reasoning for each score (be specific about evidence in the TA prediction vs the reference/code).
3) State what you believe the IDEAL answer should contain (key points/conditions).

Respond ONLY with valid JSON in EXACTLY this format:

{{
  "scores": {{
    "accuracy": 1-3,
    "completeness": 1-3,
    "relevance": 1-3,
    "clarity": 1-3
  }},
  "reasoning": {{
    "accuracy": "...",
    "completeness": "...",
    "relevance": "...",
    "clarity": "..."
  }},
  "ideal_answer_belief": "..."
}}
""".strip()


def build_student_explain_user_prompt(code: str, question: str, reference: str, prediction: str) -> str:
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


# Turn 2: teacher independently scores, compares, ToM-diagnoses, guides (JSON)
TEACHER_TOM_SYSTEM_PROMPT = f"""
You are a 30B teacher LLM supervising a student LLM-as-judge.

You will receive:
- code, question, reference answer, TA prediction
- the student judge's JSON containing:
  - scores
  - reasoning
  - ideal_answer_belief

Use the SAME anchored rubric the student is supposed to use:

{ANCHORED_RUBRIC}

YOUR TASK (multi-step):
1) Independent evaluation:
   Compute YOUR OWN scores (1-3) for accuracy, completeness, relevance, clarity using the anchored rubric.
   Base your judgment strictly on the given code/question/reference/prediction.

2) Compare:
   Compare your scores to the student's scores and identify disagreements.

3) Theory-of-mind diagnosis:
   Infer the student's mental model:
   - what it believed mattered most,
   - what evidence it attended to,
   - what it overlooked or misinterpreted,
   - what assumptions/heuristics drove its scoring.
   Explicitly connect: beliefs -> attention -> reasoning -> scores.

4) Guidance:
   Provide targeted mentoring to update the student's mental model and evaluation strategy.
   (Do not only correct the outcome; improve how it reasons with the rubric.)

5) Demonstration:
   Provide a short example of an improved student judgment (how the student SHOULD score + brief justification).

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
  "student_mental_model": "...",
  "theory_of_mind_guidance": "...",
  "improved_judgment_example": "..."
}}
""".strip()


def build_teacher_tom_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    student_json: Dict[str, Any],
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

Student Judge Output (JSON):
{json.dumps(student_json, indent=2, ensure_ascii=False)}
""".strip()


# Turn 3: student re-scores after teacher guidance (strict scores JSON, no reasoning)
STUDENT_RESCORE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing a TA answer.

You will receive the same code/question/reference/prediction plus:
- your prior scores
- teacher feedback and guidance

Use the SAME anchored rubric:

{ANCHORED_RUBRIC}

TASK:
Re-score the TA prediction on:
accuracy, completeness, relevance, clarity (integers 1-3).

Return ONLY valid JSON in EXACTLY this format:

{{
  "accuracy": {{"score": 1-3}},
  "completeness": {{"score": 1-3}},
  "relevance": {{"score": 1-3}},
  "clarity": {{"score": 1-3}}
}}
""".strip()


def build_student_rescore_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    prior_scores: Dict[str, int],
    teacher_guidance_json: Dict[str, Any],
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

Your Prior Scores:
{json.dumps(prior_scores, indent=2)}

Teacher Feedback (JSON):
{json.dumps(teacher_guidance_json, indent=2, ensure_ascii=False)}

Now re-score using the anchored rubric and return ONLY the strict score JSON.
""".strip()


# -----------------------------
# Parsing helpers
# -----------------------------
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _extract_first_json_obj(text: str) -> Optional[str]:
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def _safe_load_json(text: str) -> Optional[Dict[str, Any]]:
    js_text = _extract_first_json_obj(text)
    if not js_text:
        return None
    try:
        return json.loads(js_text)
    except Exception:
        return None


def parse_student_explain(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Expected:
    {
      "scores": {...},
      "reasoning": {...},
      "ideal_answer_belief": "..."
    }
    """
    obj = _safe_load_json(raw_text)
    if not isinstance(obj, dict):
        return None

    scores = obj.get("scores")
    reasoning = obj.get("reasoning")
    ideal = obj.get("ideal_answer_belief")

    if not isinstance(scores, dict) or not isinstance(reasoning, dict) or not isinstance(ideal, str):
        return None

    out_scores: Dict[str, int] = {}
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = scores.get(k)
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        if isinstance(v, float):
            v = int(round(v))
        if not isinstance(v, int) or v < 1 or v > 3:
            return None
        out_scores[k] = v

        rv = reasoning.get(k)
        if not isinstance(rv, str):
            return None

    return {
        "scores": out_scores,
        "reasoning": {k: str(reasoning[k]) for k in ["accuracy", "completeness", "relevance", "clarity"]},
        "ideal_answer_belief": ideal.strip(),
    }


def parse_teacher_tom(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Expected teacher JSON with teacher_scores, score_comparison, student_mental_model,
    theory_of_mind_guidance, improved_judgment_example.
    """
    obj = _safe_load_json(raw_text)
    if not isinstance(obj, dict):
        return None

    teacher_scores = obj.get("teacher_scores")
    comparison = obj.get("score_comparison")
    mental = obj.get("student_mental_model")
    guidance = obj.get("theory_of_mind_guidance")
    example = obj.get("improved_judgment_example")

    if not isinstance(teacher_scores, dict) or not isinstance(comparison, dict):
        return None
    if not all(isinstance(x, str) for x in [mental, guidance, example]):
        return None

    out_teacher_scores: Dict[str, int] = {}
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = teacher_scores.get(k)
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        if isinstance(v, float):
            v = int(round(v))
        if not isinstance(v, int) or v < 1 or v > 3:
            return None
        out_teacher_scores[k] = v

    return {
        "teacher_scores": out_teacher_scores,
        "score_comparison": comparison,
        "student_mental_model": mental.strip(),
        "theory_of_mind_guidance": guidance.strip(),
        "improved_judgment_example": example.strip(),
    }


def parse_strict_score_json(raw_text: str) -> Optional[Dict[str, int]]:
    """
    Strict format:
    {"accuracy":{"score":1-3}, ...}
    returns {"accuracy":1, ...}
    """
    obj = _safe_load_json(raw_text)
    if not isinstance(obj, dict):
        return None
    out: Dict[str, int] = {}
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = obj.get(k, {})
        score = v.get("score", None) if isinstance(v, dict) else None
        if isinstance(score, str) and score.isdigit():
            score = int(score)
        if isinstance(score, float):
            score = int(round(score))
        if not isinstance(score, int) or score < 1 or score > 3:
            return None
        out[k] = score
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    random.seed(RANDOM_SEED)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Models
    student = HFChat(STUDENT_JUDGE_MODEL, temperature=0.0, max_new_tokens=900)  # extra space for rubric+reasoning
    teacher = HFChat(TEACHER_MODEL, temperature=0.0, max_new_tokens=1100)       # extra space for rubric+ToM guidance

    n = len(data)
    k = int(round(TEACHER_INTERVENTION_RATE * n))
    intervene_ids = set(random.sample(range(n), k)) if n > 0 and k > 0 else set()

    records = []

    for idx, ex in enumerate(data):
        code = ex.get("code", "")
        question = ex.get("question", "")
        reference = ex.get("answer", "")
        prediction = ex.get("prediction", "")

        # --- Turn 1: Student scores + reasoning + ideal answer belief
        s_msgs = [
            {"role": "system", "content": STUDENT_EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": build_student_explain_user_prompt(code, question, reference, prediction)},
        ]
        raw_student_explain = student.invoke(s_msgs)
        student_explain = parse_student_explain(raw_student_explain)

        base_scores: Optional[Dict[str, int]] = student_explain["scores"] if student_explain else None

        # --- Turn 2: Teacher intervention (random subset)
        teacher_intervened = idx in intervene_ids
        raw_teacher = None
        teacher_tom = None
        teacher_scores = None

        # --- Turn 3: Student re-score after guidance
        raw_student_rescore = None
        rescored = None

        if teacher_intervened and student_explain is not None:
            t_msgs = [
                {"role": "system", "content": TEACHER_TOM_SYSTEM_PROMPT},
                {"role": "user", "content": build_teacher_tom_user_prompt(code, question, reference, prediction, student_explain)},
            ]
            raw_teacher = teacher.invoke(t_msgs)
            teacher_tom = parse_teacher_tom(raw_teacher)
            teacher_scores = teacher_tom["teacher_scores"] if teacher_tom else None

            # If teacher produced guidance, let student re-score once
            if teacher_tom is not None and base_scores is not None:
                r_msgs = [
                    {"role": "system", "content": STUDENT_RESCORE_SYSTEM_PROMPT},
                    {"role": "user", "content": build_student_rescore_user_prompt(
                        code, question, reference, prediction, base_scores, teacher_tom
                    )},
                ]
                raw_student_rescore = student.invoke(r_msgs)
                rescored = parse_strict_score_json(raw_student_rescore)

        # Final scores used in outputs
        final_scores = rescored if rescored is not None else base_scores

        record = {
            "id": ex.get("id", idx),
            "code": code,
            "question": question,
            "answer": reference,
            "prediction": prediction,

            "teacher_intervened": teacher_intervened,

            # Student turn 1
            "raw_student_explain_output": raw_student_explain,
            "student_explain_json": student_explain,
            "initial_student_scores": base_scores,

            # Teacher turn 2
            "raw_teacher_output": raw_teacher,
            "teacher_tom_json": teacher_tom,
            "teacher_scores": teacher_scores,

            # Student turn 3
            "raw_student_rescore_output": raw_student_rescore,
            "rescored_student_scores": rescored,

            # Flattened final columns for CSV
            "accuracy": final_scores.get("accuracy") if final_scores else None,
            "completeness": final_scores.get("completeness") if final_scores else None,
            "relevance": final_scores.get("relevance") if final_scores else None,
            "clarity": final_scores.get("clarity") if final_scores else None,

            # initial vs rescored
            "accuracy_initial": base_scores.get("accuracy") if base_scores else None,
            "completeness_initial": base_scores.get("completeness") if base_scores else None,
            "relevance_initial": base_scores.get("relevance") if base_scores else None,
            "clarity_initial": base_scores.get("clarity") if base_scores else None,

            "accuracy_rescored": rescored.get("accuracy") if rescored else None,
            "completeness_rescored": rescored.get("completeness") if rescored else None,
            "relevance_rescored": rescored.get("relevance") if rescored else None,
            "clarity_rescored": rescored.get("clarity") if rescored else None,
        }

        records.append(record)

    # write JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # write CSV
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
