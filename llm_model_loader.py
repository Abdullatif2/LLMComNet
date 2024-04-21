import gc
import transformers
import torch


# Set preferred encoding right at the beginning to ensure all file operations use UTF-8
import locale
locale.getpreferredencoding = lambda: "UTF-8"


def initialize_system():
    """Initializes system by clearing CUDA cache."""
    torch.cuda.empty_cache()
    print("CUDA cache cleared")

def load_transformer_model(model_name: str):
    """Loads a transformer model and tokenizer, adjusting for available GPU resources."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to("cuda:0")
    model.eval()
    return tokenizer, model

def cleanup_resources(*variables):
    """Deletes specified global variables and collects garbage."""
    for var in variables:
        if var in globals():
            del globals()[var]
    gc.collect()
    initialize_system()
    