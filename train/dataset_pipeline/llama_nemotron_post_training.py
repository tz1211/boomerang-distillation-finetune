from datasets import load_dataset, interleave_datasets
from transformers import PreTrainedTokenizer
from typing import Dict, Any


def tokenize_instruction_following(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """
    Tokenize instruction-following examples with chat template.
    Prompt tokens are labeled -100, response tokens are labeled with actual token ids.
    
    Args:
        example: Dataset example with 'input' and 'output' fields
            - input: list of dicts with 'role' and 'content' (e.g., [{"role": "user", "content": "..."}])
            - output: string with assistant response
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Tokenized example with 'input_ids', 'attention_mask', and 'labels' where
        prompt tokens in labels are set to -100.
    """
    # Extract user message from input
    user_content = example["input"][0]["content"] if isinstance(example["input"], list) and len(example["input"]) > 0 else ""
    assistant_content = example["output"]
    
    # Create messages format for chat template
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False # Don't add generation prompt during training
    )
    
    # Tokenize the full sequence
    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,  # Return as lists, not tensors
    )
    
    labels = tokenized["input_ids"].copy()
    
    # Determine prompt length and mask prompt tokens
    if len(messages) > 1 and messages[-1].get("role") == "assistant":
        # Get prompt messages (all except the last assistant message)
        prompt_messages = messages[:-1]
        
        # Apply chat template to prompt only (with generation prompt to match format)
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompt_tokenized = tokenizer(prompt_text, truncation=False, add_special_tokens=True)
        prompt_length = len(prompt_tokenized["input_ids"])
        
        # Mask prompt tokens (set to -100) - use prompt length directly
        mask_length = min(prompt_length, len(labels))
        for i in range(mask_length):
            labels[i] = -100
    
    # Also mask padding tokens (set to -100)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for i in range(len(labels)):
        if labels[i] == pad_token_id:
            labels[i] = -100
    
    # Add labels to tokenized output
    tokenized["labels"] = labels
    
    return tokenized


def prepare_llama_nemotron_post_training(tokenizer: PreTrainedTokenizer, max_length: int, seed: int):
    # Load each split separately
    code_dataset = load_dataset(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        split="code",
        streaming=True,
    )
    math_dataset = load_dataset(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        split="math",
        streaming=True,
    )
    science_dataset = load_dataset(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        split="science",
        streaming=True,
    )
    
    # Interleave the datasets
    train_dataset = interleave_datasets([code_dataset, math_dataset, science_dataset])
    
    test = train_dataset.take(512)
    
    # Create tokenization function
    def tokenize_function(example):
        return tokenize_instruction_following(
            example=example,
            tokenizer=tokenizer,
            max_length=max_length,
        )
    
    # Tokenize the dataset
    tokenized_train = train_dataset.shuffle(buffer_size=100_000, seed=seed).map(
        tokenize_function, batched=False
    )
    eval_dataset = test.map(tokenize_function, batched=False)
    
    return tokenized_train, eval_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    max_length = 2048
    seed = 0
    tokenized_train, eval_dataset = prepare_llama_nemotron_post_training(tokenizer, max_length, seed)
    print("dataset loaded and tokenized")
    # Get a sample to debug
    sample = next(iter(eval_dataset))
    
    # Extract original response
    original_response = sample["output"]
    
    # Extract tokens that are NOT masked (labels != -100)
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    
    # Get response token IDs (where labels != -100)
    response_token_ids = [input_ids[i] for i in range(len(labels)) if labels[i] != -100]
    
    # Decode the response tokens
    decoded_response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
    
    print("\n" + "="*80)
    print("ORIGINAL RESPONSE:")
    print("="*80)
    print(original_response)
    print("\n" + "="*80)
    print("DECODED FROM UNMASKED TOKENS:")
    print("="*80)
    print(decoded_response)
    print("\n" + "="*80)
    print("COMPARISON:")
    print("="*80)
    print(f"Original length: {len(original_response)}")
    print(f"Decoded length: {len(decoded_response)}")
    print(f"Match: {original_response.strip() == decoded_response.strip()}")
    
    # Also show the token IDs for debugging
    print(f"\nResponse token IDs (first 20): {response_token_ids[:20]}")
    print(f"Total response tokens: {len(response_token_ids)}")
    print(f"Total masked tokens: {sum(1 for l in labels if l == -100)}")
    