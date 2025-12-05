from datasets import load_dataset
from transformers import PreTrainedTokenizer

def prepare_the_pile(tokenizer: PreTrainedTokenizer, max_length: int, seed: int):
    train_dataset = load_dataset(
        "EleutherAI/the_pile_deduplicated",
        split="train",
        streaming=True,
    )

    test = train_dataset.take(512)
    # Note: skipping the first 50k examples for reproducibility. This does not affect functionality.
    train_dataset = train_dataset.skip(50000)

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    tokenized_train = train_dataset.shuffle(buffer_size=100_000, seed=seed).map(
        tokenize_function, batched=True
    )
    eval_dataset = test.map(tokenize_function, batched=True)

    return tokenized_train, eval_dataset