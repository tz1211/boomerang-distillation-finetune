"""MATH500 evaluation - Hendrycks MATH dataset (500 subset)."""

from typing import Dict, Optional

from vllm import LLM
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluate.utils.utils import batch_generate, grade_answer


def evaluate_math500(
    model: LLM,
    tokenizer: AutoTokenizer,
    apply_chat_template: bool = True,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    limit: Optional[int] = None,
    system_prompt: Optional[str] = None,
    log_n_results: int = 100,
) -> Dict:
    """
    Evaluate model on MATH500 dataset (subset of Hendrycks MATH).
    
    Args:
        model: The vLLM LLM model
        tokenizer: The tokenizer
        apply_chat_template: Whether to apply chat template
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        limit: Limit number of examples (default 500 for MATH500)
        system_prompt: Optional system prompt to prepend to each problem
        log_n_results: Number of results to log
    Returns:
        Dictionary with accuracy and detailed results
    """
    print("Loading MATH dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test") # The original hendrycks/competition_math dataset is taken down

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    problems = [item["problem"] for item in dataset]
    gt_answers = [item["answer"] for item in dataset]
    levels = [item["level"] for item in dataset]
    
    print(f"Evaluating on {len(problems)} MATH problems...")
    
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
    
    # Evaluate answers 
    correct = 0
    results = []
    level_stats = {}
    
    for i, (response, gt_answer) in enumerate(zip(responses, gt_answers)):
        is_correct = grade_answer(response, gt_answer, extract=True)

        # Update overall statistics
        if is_correct:
            correct += 1
        
        # Update level statistics
        level = levels[i]
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        if is_correct:
            level_stats[level]["correct"] += 1
        
        results.append({
            "problem": problems[i],
            "formatted_problem": formatted_problems[i],
            "ground_truth": gt_answer,
            "level": level,
            "raw_response": response,
            "correct": is_correct,
        })
    
    # Calculate accuracies
    accuracy = correct / len(results) if results else 0.0
    
    # Calculate accuracy for each level
    by_level = {}
    for level, stats in level_stats.items():
        by_level[level] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }
    by_level = dict(sorted(by_level.items()))
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "samples": results[:log_n_results],
        "by_level": by_level,
    }
