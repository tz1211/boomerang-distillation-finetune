"""GSM8K evaluation - Grade School Math 8K problems."""

from typing import Dict, Optional

from vllm import LLM
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluate.tasks.utils import batch_generate, extract_answer_boxed


def evaluate_gsm8k(
    model: LLM,
    tokenizer: AutoTokenizer,
    apply_chat_template: bool = True,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    limit: Optional[int] = None,
    system_prompt: Optional[str] = None,
    log_n_results: int = 100,
) -> Dict:
    """
    Evaluate model on GSM8K dataset.
    
    Args:
        model: The vLLM LLM model
        tokenizer: The tokenizer
        apply_chat_template: Whether to apply chat template
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        limit: Limit number of examples
        system_prompt: Optional system prompt to prepend to each question
        log_n_results: Number of results to log
    Returns:
        Dictionary with accuracy and detailed results
    """
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    questions = [item["question"] for item in dataset]
    ground_truth_answers = [item["answer"] for item in dataset]
    
    # Extract numeric answers from ground truth
    def extract_gt_answer(answer_text: str) -> str:
        # GSM8K answers are in format "#### 123" or just the number
        match = answer_text.split("####")
        if len(match) > 1:
            return match[-1].strip()
        return answer_text.strip()
    
    gt_answers = [int(extract_gt_answer(ans)) for ans in ground_truth_answers]
    
    print(f"Evaluating on {len(questions)} GSM8K problems...")
    
    # Generate responses
    formatted_problems, responses = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=questions,
        apply_chat_template=apply_chat_template,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )
    
    # Evaluate answers
    correct = 0
    results = []
    
    for i, (response, gt_answer) in enumerate(zip(responses, gt_answers)):
        # Extract answer from response
        predicted_answer = extract_answer_boxed(response)
        if predicted_answer is not None:
            try:
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
            "problem": questions[i],
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
