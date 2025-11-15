"""AIME evaluation - American Invitational Mathematics Examination."""

from typing import Dict, Optional

from vllm import LLM
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluate.tasks.utils import batch_generate, extract_answer_boxed


def evaluate_aime(
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
    Evaluate model on AIME dataset.
    
    Note: AIME dataset may need to be loaded from a specific source.
    This implementation assumes it's available via HuggingFace datasets.
    
    Args:
        model: The vLLM LLM model
        tokenizer: The tokenizer
        apply_chat_template: Whether to apply chat template
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        limit: Limit number of examples
        system_prompt: Optional system prompt to prepend to each problem
        log_n_results: Number of results to log
    Returns:
        Dictionary with accuracy and detailed results
    """
    print("Loading AIME dataset...")
    
    # Load AIME dataset 
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    # Extract problems and answers
    problems = [item["problem"] for item in dataset]
    ground_truth_answers = [int(item["answer"]) for item in dataset]
    
    print(f"Evaluating on {len(problems)} AIME problems...")
    
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
    
    for i, (response, gt_answer) in enumerate(zip(responses, ground_truth_answers)):
        # Extract answer from response
        predicted_answer = extract_answer_boxed(response)
        if predicted_answer is not None:
            try:
                predicted_answer = predicted_answer.replace(',', '')
                predicted_answer = int(predicted_answer)
            except ValueError: # If the model can't follow instruction and return only a number inside \boxed{}, we treat it as incorrect
                predicted_answer = None
        
        # Check correctness 
        is_correct = False
        if predicted_answer:
            is_correct = predicted_answer == gt_answer
        
        if is_correct:
            correct += 1
        
        results.append({
            "problem": problems[i],
            "formatted_problem": formatted_problems[i],
            "ground_truth": gt_answer,
            "raw_response": response,
            "filtered_answer": predicted_answer,
            "correct": is_correct,
        })
    
    accuracy = correct / len(results) if results else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "samples": results[:log_n_results],
    }
