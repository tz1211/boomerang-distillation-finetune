"""IFEval evaluation - Instruction Following Evaluation."""

from typing import Dict, Optional

from vllm import LLM
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluate.utils.utils import batch_generate, grade_instruction_following
from evaluate.utils.instructions_util import IFEvalInputItem

def evaluate_ifeval(
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
    Evaluate model on IFEval dataset (instruction following evaluation).
    
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
    print("Loading IFEval dataset...")
    
    dataset = load_dataset("google/IFEval", split="train")
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    prompts = []
    input_items = []  # Store InputItem objects for evaluation
    
    for item in dataset:
        prompt = item.get("prompt", "")
        instruction_id_list = item.get("instruction_id_list", [])
        kwargs_list = item.get("kwargs", [])
        
        # Create InputItem for grade_instruction_following
        input_item = IFEvalInputItem(prompt, instruction_id_list, kwargs_list)
        prompts.append(prompt)
        input_items.append(input_item)
    
    # Generate responses
    formatted_prompts, responses = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        apply_chat_template=apply_chat_template,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )
    
    # Evaluate responses using grade_instruction_following
    correct = 0
    results = []
    
    for i, (response, input_item) in enumerate(zip(responses, input_items)):
        response_stripped = response.split("</think>")[-1].strip() # Only evaluate the response after the </think> tag
        instruction_results = grade_instruction_following(input_item, response_stripped)
        
        if all(instruction_results.values()):
            correct += 1
        
        results.append({
            "prompt": prompts[i],
            "formatted_prompt": formatted_prompts[i],
            "instruction_results": instruction_results,
            "raw_response": response,
            "correct": all(instruction_results.values()),
        })
    
    accuracy = correct / len(results) if results else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "samples": results[:log_n_results],
    }

