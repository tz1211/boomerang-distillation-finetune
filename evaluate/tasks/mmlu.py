"""MMLU evaluation - Massive Multitask Language Understanding."""

from typing import Dict, List, Optional

from vllm import LLM
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluate.tasks.utils import batch_generate, extract_answer_boxed


def evaluate_mmlu(
    model: LLM,
    tokenizer: AutoTokenizer,
    apply_chat_template: bool = True,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    limit: Optional[int] = None,
    subjects: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    log_n_results: int = 100,
) -> Dict:
    """
    Evaluate model on MMLU dataset (multiple choice questions).
    
    Args:
        model: The vLLM LLM model
        tokenizer: The tokenizer
        apply_chat_template: Whether to apply chat template
        max_new_tokens: Maximum tokens to generate (should be small for MC)
        temperature: Sampling temperature
        limit: Limit number of examples per subject
        subjects: List of subjects to evaluate on (None = all)
        system_prompt: Optional system prompt to prepend to each question
        log_n_results: Number of results to log
    Returns:
        Dictionary with accuracy and detailed results
    """
    print("Loading MMLU dataset...")
    
    # MMLU has multiple subjects
    if subjects is None:
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_physics",
            "computer_security", "conceptual_physics", "econometrics",
            "electrical_engineering", "elementary_mathematics", "formal_logic",
            "global_facts", "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_european_history",
            "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics",
            "high_school_microeconomics", "high_school_physics",
            "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history",
            "human_aging", "human_sexuality", "international_law",
            "jurisprudence", "logical_fallacies", "machine_learning",
            "management", "marketing", "medical_genetics", "miscellaneous",
            "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
            "prehistory", "professional_accounting", "professional_law",
            "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions"
        ]
    
    all_results = {}
    total_correct = 0
    total_questions = 0
    
    for subject in subjects:
        try:
            print(f"Evaluating MMLU subject: {subject}...")
            dataset = load_dataset("cais/mmlu", subject, split="test")
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            problems = []
            correct_answers = []
            
            for item in dataset:
                problem = item["question"]
                choices = item["choices"]
                correct_answer = item["answer"]
                
                # Format as multiple choice question
                input_prompt = f"{problem.strip()}\nA. {choices[0].strip()}\nB. {choices[1].strip()}\nC. {choices[2].strip()}\nD. {choices[3].strip()}\nAnswer:"
                
                problems.append(input_prompt)
                correct_answers.append(correct_answer)
            
            # Generate responses
            formatted_problems, responses = batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=problems,
                apply_chat_template=apply_chat_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )
            
            # Evaluate
            subject_correct = 0
            subject_results = []
            
            for i, (response, correct_answer) in enumerate(zip(responses, correct_answers)):
                # Extract answer choice (A, B, C, or D)
                predicted_answer = extract_answer_boxed(response)
                
                is_correct = predicted_answer == correct_answer
                if is_correct:
                    subject_correct += 1
                    total_correct += 1
                
                total_questions += 1
                
                subject_results.append({
                    "problem": problems[i],
                    "formatted_problem": formatted_problems[i],
                    "ground_truth": correct_answer,
                    "raw_response": response,
                    "filtered_answer": predicted_answer,
                    "correct": is_correct,
                })
            
            subject_accuracy = subject_correct / len(subject_results) if subject_results else 0.0
            
            all_results[subject] = {
                "accuracy": subject_accuracy,
                "correct": subject_correct,
                "total": len(subject_results),
                "samples": subject_results[:log_n_results],
            }
            
            print(f"{subject}: {subject_accuracy:.4f} ({subject_correct}/{len(subject_results)})")
            
        except Exception as e:
            print(f"Error evaluating {subject}: {e}")
            all_results[subject] = {"error": str(e)}
    
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
    
    return {
        "accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_questions,
        "by_subject": all_results,
    }


