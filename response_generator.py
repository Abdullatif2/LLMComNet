
import xml.etree.ElementTree as ET
import torch
from typing import List
from prompt_setup_utilities import setup_prompt_with_functions
from communication_system_models import *
def generate_response(prompt, model, tokenizer, generation_config_overrides={}):
    """Generates a response from the model based on the given prompt and optional configuration overrides."""
    underlying_model = model.module if hasattr(model, 'module') else model
    prompt = setup_prompt_with_functions(prompt, BandwidthOptimization, BeamPrediction, ChannelEstimation, PowerAllocation, SettingRecommendation, Summation)

    # Setup prompt with predefined functions

    # Prepare generation configurations
    generation_config = {
        "use_cache": True,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": 512,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,  # Ensure to use pad_token_id
        **generation_config_overrides
    }
    inputs = tokenizer(prompt, return_tensors="pt").to(underlying_model.device)
    n_tokens = inputs.input_ids.numel()

    with torch.inference_mode():
        generated_tokens = underlying_model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, **generation_config)
    
    return tokenizer.decode(generated_tokens.squeeze()[n_tokens:], skip_special_tokens=False)




