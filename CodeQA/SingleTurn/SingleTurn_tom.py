
import json
import random
import re
from typing import Any, Dict, Optional, Tuple, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM



INPUT_JSON = "mini.json"                         # expects list of dicts with keys: code, question, answer, prediction
OUTPUT_JSON = "rq3_single_mini.json"
OUTPUT_CSV = "rq3_single_mini.csv"


STUDENT_JUDGE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

TEACHER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Teacher intervention rate (30% of examples)
TEACHER_INTERVENTION_RATE = 0.30

# Reproducibility
RANDOM_SEED = 42
# =========================================================


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
        """
        chat_messages: [{"role":"system"/"user", "content": "..."}]
        """
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

        # Extract generated tokens
        gen_tokens = out[0, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return text


# -----------------------------
# PROMPTS Student (judge)
# -----------------------------
JUDGE_SYSTEM_PROMPT = """
You are a large language model acting as a judge for assessing the performance of a Teaching Assistant (TA) in an introductory Python programming course.

The TA is an LLM that answers student questions about Python code. Your job is to evaluate the quality of the TA's answer.

You will receive:
- A Python code snippet
- A student question about that code
- A reference (correct) answer
- A TA LLM-generated answer (called the prediction)

Your task is to evaluate how well the TA's prediction answers the student's question, using the following four dimensions. For each, provide:
- An integer score from 1 to 3



### Accuracy
Compare the prediction with the reference to assess factual correctness and understanding of the code’s behavior and intent.
You must judge whether the prediction reflects accurate behavior and matches core facts from the reference. 
You need to consider semantic meaning of code comprehension: understanding the structure, functionality, and intent behind the code.\n"

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

Code:
```python
def count_even(nums):
    total = 0
    for x in nums:
        if x % 2 == 0:
            total += 1
    return total
Question: What does this function return when given a list of integers?
Reference Answer: It returns the count of even numbers in the list.
Prediction: It returns the count of odd numbers in the list.

Evaluation Output:
{

"accuracy": { "score": 1 },
"completeness": { "score": 1 },
"relevance": { "score": 2 },
"clarity": { "score": 3 }

}

Final Instructions:
For the given input (code, question, reference answer, and prediction), evaluate the prediction on the four metrics defined above.
Base your evaluation strictly on the content provided. Do not hallucinate missing information. Be consistent and objective.
Do not include reasoning or explanations.

Respond only with a JSON object in the exact format:
{
"accuracy": { "score": 1-3},
"completeness": {"score": 1-3},
"relevance": {"score": 1-3},
"clarity": {"score": 1-3}
}
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
# Teacher intervention prompt (from notebook pattern)
# -----------------------------
TEACHER_SYSTEM_PROMPT = """
You are a 70B teacher LLM model reviewing a student LLM-as-judge's evaluation of a Teaching Assistant's (TA) answer.

You will receive:
- Python code snippet
- Student's question
- Reference (correct) answer
- TA's predicted answer
- Scores assigned by the student LLM-as-judge

Your task:
- Examine the TA's predicted answer in context of the code, question, and reference.
- For any dimension (Accuracy, Completeness, Relevance, Clarity) where the score is less than 3,
  provide clear, concise feedback (2–4 sentences) explaining what could be improved.
- If a dimension has no issues, do not include it in your response.

Respond ONLY with a JSON object where keys are the dimension names (lowercase)
and values are the feedback strings.

Example output:
{
  "accuracy": "The prediction misrepresents the function’s return value.",
  "clarity": "The explanation lacks structure and is hard to follow."
}

Rubric:

### Accuracy
- 1: Completely incorrect or irrelevant.
- 2: Partially correct but with major mistakes or omissions.
- 3: Fully correct and matches the reference.

### Completeness
- 1: Omits most key information.
- 2: Covers some but misses important parts.
- 3: Fully covers all essential information.

### Relevance
- 1: Irrelevant or mostly unrelated.
- 2: Partially related but misses main point.
- 3: Fully focused and directly addresses the question.

### Clarity
- 1: Confusing, vague, or incoherent.
- 2: Understandable but awkwardly phrased or unclear.
- 3: Clear, concise, and easy to understand.
""".strip()


def build_teacher_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    scores: Dict[str, int],
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

Student Judge Scores (1-3):
{json.dumps(scores, indent=2)}
""".strip()


# -----------------------------
# student reflects + revises (single turn)
# -----------------------------
REVISION_SYSTEM_PROMPT = JUDGE_SYSTEM_PROMPT  # keep the same judging rubric/prompt


def build_revision_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    original_scores: Dict[str, int],
    critiques: Dict[str, str],
) -> str:
    critique_text = "\n".join(
        f"{dim.upper()} Feedback: {txt.strip()}" for dim, txt in critiques.items()
    )

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

Original Scores:
{json.dumps(original_scores, indent=2)}

Teacher Feedback:
{critique_text}

Now re-score the prediction using the same rubric and return ONLY valid JSON in the same format.
""".strip()


# -----------------------------
# Parsing helpers
# -----------------------------
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _extract_first_json_obj(text: str) -> Optional[str]:
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def parse_judge_scores(raw_text: str) -> Optional[Dict[str, int]]:
    """
    Accepts the strict format:
    {"accuracy":{"score":1-3}, ...}
    Returns: {"accuracy": int, ...} or None if parsing fails.
    """
    js_text = _extract_first_json_obj(raw_text)
    if not js_text:
        return None
    try:
        obj = json.loads(js_text)
        out: Dict[str, int] = {}
        for k in ["accuracy", "completeness", "relevance", "clarity"]:
            v = obj.get(k, {})
            score = v.get("score", None) if isinstance(v, dict) else None
            if isinstance(score, str) and score.isdigit():
                score = int(score)
            if isinstance(score, float):
                score = int(round(score))
            if not isinstance(score, int):
                return None
            out[k] = score
        return out
    except Exception:
        return None


def parse_teacher_feedback(raw_text: str) -> Optional[Dict[str, str]]:
    js_text = _extract_first_json_obj(raw_text)
    if not js_text:
        return None
    try:
        obj = json.loads(js_text)
        out: Dict[str, str] = {}
        for k in ["accuracy", "completeness", "relevance", "clarity"]:
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                out[k] = obj[k].strip()
        return out
    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    random.seed(RANDOM_SEED)

    # Load data
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Init models
    student = HFChat(STUDENT_JUDGE_MODEL, temperature=0.0, max_new_tokens=256)
    teacher = HFChat(TEACHER_MODEL, temperature=0.0, max_new_tokens=384)

    n = len(data)
    k = int(round(TEACHER_INTERVENTION_RATE * n))
    intervene_ids = set(random.sample(range(n), k)) if n > 0 and k > 0 else set()

    records = []

    for idx, ex in enumerate(data):
        code = ex.get("code", "")
        question = ex.get("question", "")
        answer = ex.get("answer", "")  # reference
        prediction = ex.get("prediction", "")

        # Student judge scoring
        judge_msgs = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(code, question, answer, prediction)},
        ]
        raw_judge = student.invoke(judge_msgs)
        base_scores = parse_judge_scores(raw_judge)

        # Teacher intervention (random 30%)
        teacher_intervened = idx in intervene_ids
        teacher_feedback = None
        revised_scores = None
        raw_teacher = None
        raw_revised = None

        if teacher_intervened and base_scores is not None:
            t_msgs = [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": build_teacher_user_prompt(code, question, answer, prediction, base_scores)},
            ]
            raw_teacher = teacher.invoke(t_msgs)
            teacher_feedback = parse_teacher_feedback(raw_teacher) or {}

            # If teacher provided any critiques, let student re-score once
            if teacher_feedback:
                r_msgs = [
                    {"role": "system", "content": REVISION_SYSTEM_PROMPT},
                    {"role": "user", "content": build_revision_user_prompt(code, question, answer, prediction, base_scores, teacher_feedback)},
                ]
                raw_revised = student.invoke(r_msgs)
                revised_scores = parse_judge_scores(raw_revised)

        # Final scores used for CSV
        final_scores = revised_scores if revised_scores is not None else base_scores

        record = {
            "id": ex.get("id", idx),
            "code": code,
            "question": question,
            "answer": answer,
            "prediction": prediction,

            "initial_student_scores": base_scores,
            "raw_student_judge_output": raw_judge,

            "teacher_intervened": teacher_intervened,
            "teacher_feedback": teacher_feedback,
            "raw_teacher_output": raw_teacher,

            "improved_student_scores": revised_scores,
            "raw_improved_output": raw_revised,

            
# Flattened final columns (used for CSV compatibility)
"accuracy": final_scores.get("accuracy") if final_scores else None,
"completeness": final_scores.get("completeness") if final_scores else None,
"relevance": final_scores.get("relevance") if final_scores else None,
"clarity": final_scores.get("clarity") if final_scores else None,

# Also keep initial vs improved scores explicitly
"accuracy_initial": base_scores.get("accuracy") if base_scores else None,
"completeness_initial": base_scores.get("completeness") if base_scores else None,
"relevance_initial": base_scores.get("relevance") if base_scores else None,
"clarity_initial": base_scores.get("clarity") if base_scores else None,

"accuracy_improved": (revised_scores.get("accuracy") if revised_scores else None),
"completeness_improved": (revised_scores.get("completeness") if revised_scores else None),
"relevance_improved": (revised_scores.get("relevance") if revised_scores else None),
"clarity_improved": (revised_scores.get("clarity") if revised_scores else None),
        }
        records.append(record)

    # Write JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Write CSV (wide format like your screenshot)
    df = pd.DataFrame(records)
    df_out = df[[
        "code", "question", "answer", "prediction", "accuracy", "completeness", "relevance", "clarity", "accuracy_initial", "completeness_initial", "relevance_initial", "clarity_initial", "accuracy_improved", "completeness_improved", "relevance_improved", "clarity_improved"
    ]]
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("Done")
    print("JSON:", OUTPUT_JSON)
    print("CSV :", OUTPUT_CSV)
    print(f"Teacher intervened on {len(intervene_ids)}/{n} examples ({TEACHER_INTERVENTION_RATE*100:.0f}%).")


if __name__ == "__main__":
    main()
