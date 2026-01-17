#!/usr/bin/env python3
"""
Single-turn Theory-of-Mind Supervision (ToMS) Evaluation
Adapted for LOCAL MACHINE execution with automatic GPU/CPU detection

Usage:
    python local_machine_toms.py
    python local_machine_toms.py --input my_data.json --output results.json
    python local_machine_toms.py --cache-dir ./model_cache
"""

import json
import random
import re
import os
import argparse
from typing import Any, Dict, Optional, List
import sys

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =============================
# Default Configuration
# =============================
DEFAULT_INPUT_JSON = "mini.json"
DEFAULT_OUTPUT_JSON = "miniall.json"
DEFAULT_OUTPUT_CSV = "miniall.csv"

# Original models from cluster version
DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

DEFAULT_INTERVENTION_RATE = 1.00
DEFAULT_RANDOM_SEED = 42


# =============================
# Model wrapper (HF) with auto device detection
# =============================
class HFChat:
    """HuggingFace chat model wrapper with automatic GPU/CPU detection."""
    
    def __init__(
        self, 
        model_name: str, 
        temperature: float = 0.0, 
        max_new_tokens: int = 256,
        cache_dir: Optional[str] = None,
        force_cpu: bool = False
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"{'='*60}")
        
        # Determine device
        if force_cpu:
            self.device = "cpu"
            device_map = {"": "cpu"}
            dtype = torch.float32
            print(" Forced CPU mode")
        elif torch.cuda.is_available():
            self.device = "cuda"
            device_map = "auto"
            dtype = torch.float16
            print(f" CUDA available - using GPU")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = "cpu"
            device_map = {"": "cpu"}
            dtype = torch.float32
            print(" CUDA not available - using CPU (this will be slower)")
        
        if cache_dir:
            print(f"Cache directory: {cache_dir}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Load model
        print("Loading model weights...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # More memory efficient
            )
        except Exception as e:
            print(f" Error loading model: {e}")
            print("\nTips:")
            print("  1. Try a smaller model (use --student-model and --teacher-model flags)")
            print("  2. Make sure you have enough RAM/VRAM")
            print("  3. Try --force-cpu flag if GPU causes issues")
            raise
        
        self.model.eval()
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded successfully")
        print(f"{'='*60}\n")

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
        ).to(self.device)

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
    
    def cleanup(self):
        """Clean up model resources."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
# Argument parser
# =============================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Single-turn Theory-of-Mind Supervision (ToMS) Evaluation - Local Machine Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python local_machine_toms.py
  
  # Custom input/output files
  python local_machine_toms.py --input my_data.json --output results.json
  
  # Force CPU execution
  python local_machine_toms.py --force-cpu
  
  # Custom cache directory
  python local_machine_toms.py --cache-dir ./my_cache
        """
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        default=DEFAULT_INPUT_JSON,
        help=f"Input JSON file (default: {DEFAULT_INPUT_JSON})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT_JSON})"
    )
    parser.add_argument(
        "--output-csv", 
        type=str, 
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT_CSV})"
    )
    parser.add_argument(
        "--student-model", 
        type=str, 
        default=DEFAULT_STUDENT_MODEL,
        help=f"Student judge model (default: {DEFAULT_STUDENT_MODEL})"
    )
    parser.add_argument(
        "--teacher-model", 
        type=str, 
        default=DEFAULT_TEACHER_MODEL,
        help=f"Teacher model (default: {DEFAULT_TEACHER_MODEL})"
    )
    parser.add_argument(
        "--intervention-rate", 
        type=float, 
        default=DEFAULT_INTERVENTION_RATE,
        help=f"Teacher intervention rate 0.0-1.0 (default: {DEFAULT_INTERVENTION_RATE})"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})"
    )
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default=None,
        help="HuggingFace cache directory (default: system default)"
    )
    parser.add_argument(
        "--force-cpu", 
        action="store_true",
        help="Force CPU execution even if GPU is available"
    )
    
    return parser.parse_args()


# =============================
# Main execution
# =============================
def main() -> None:
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    print("=" * 60)
    print("Single-Turn Theory-of-Mind Supervision (ToMS)")
    print("Local Machine Version")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Input file: {args.input}")
    print(f"  Output JSON: {args.output}")
    print(f"  Output CSV: {args.output_csv}")
    print(f"  Student model: {args.student_model}")
    print(f"  Teacher model: {args.teacher_model}")
    print(f"  Intervention rate: {args.intervention_rate*100:.0f}%")
    print(f"  Random seed: {args.seed}")
    if args.cache_dir:
        print(f"  Cache directory: {args.cache_dir}")
    if args.force_cpu:
        print(f"  Device: CPU (forced)")
    print()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    print(f"Loading data from: {args.input}")
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f" Loaded {len(data)} examples\n")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        print(f"Please make sure the file exists in the current directory")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f" Error: Invalid JSON in '{args.input}'")
        print(f"   {e}")
        sys.exit(1)

    # Initialize models
    print("=" * 60)
    print("Initializing models...")
    print("=" * 60)
    
    try:
        student = HFChat(
            args.student_model, 
            temperature=0.0, 
            max_new_tokens=256,
            cache_dir=args.cache_dir,
            force_cpu=args.force_cpu
        )
        
        teacher = HFChat(
            args.teacher_model, 
            temperature=0.0, 
            max_new_tokens=900,
            cache_dir=args.cache_dir,
            force_cpu=args.force_cpu
        )
    except Exception as e:
        print(f"\n Failed to initialize models")
        print(f"Error: {e}")
        sys.exit(1)

    # Determine intervention examples
    n = len(data)
    k = int(round(args.intervention_rate * n))
    intervene_ids = set(random.sample(range(n), k)) if n > 0 and k > 0 else set()
    
    print(f"Processing Configuration:")
    print(f"  Total examples: {n}")
    print(f"  Teacher interventions: {k} ({args.intervention_rate*100:.0f}%)")
    print()

    records = []

    # Process each example
    try:
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
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print(f"Processed {len(records)}/{n} examples before interruption")
        if len(records) > 0:
            save_partial = input("Save partial results? (y/n): ").strip().lower()
            if save_partial != 'y':
                print("Exiting without saving...")
                sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        print("\nCleaning up resources...")
        student.cleanup()
        teacher.cleanup()

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    # Save JSON
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON saved: {args.output}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

    # Save CSV
    try:
        df = pd.DataFrame(records)
        df_out = df[[
            "id", "code", "question", "answer", "prediction",
            "accuracy", "completeness", "relevance", "clarity",
            "accuracy_initial", "completeness_initial", "relevance_initial", "clarity_initial",
            "accuracy_rescored", "completeness_rescored", "relevance_rescored", "clarity_rescored",
            "teacher_intervened"
        ]]
        df_out.to_csv(args.output_csv, index=False, encoding="utf-8")
        print(f"✓ CSV saved: {args.output_csv}")
    except Exception as e:
        print(f" Error saving CSV: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total examples: {n}")
    print(f"Successfully processed: {len(records)}")
    print(f"Teacher interventions: {len(intervene_ids)} ({args.intervention_rate*100:.0f}%)")
    print(f"\nOutput files:")
    print(f"  - {args.output}")
    print(f"  - {args.output_csv}")
    print("\n Done!")


if __name__ == "__main__":
    main()