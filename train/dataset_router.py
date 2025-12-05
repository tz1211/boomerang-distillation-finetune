from transformers import PreTrainedTokenizer

from train.dataset_pipeline.the_pile import prepare_the_pile
from train.dataset_pipeline.llama_nemotron_post_training import prepare_llama_nemotron_post_training

def prepare_dataset(dataset_name: str, tokenizer: PreTrainedTokenizer, max_length: int, seed: int):
    if dataset_name == "EleutherAI/the_pile_deduplicated":
        return prepare_the_pile(tokenizer, max_length, seed)
    elif dataset_name == "nvidia/Llama-Nemotron-Post-Training-Dataset":
        return prepare_llama_nemotron_post_training(tokenizer, max_length, seed)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")