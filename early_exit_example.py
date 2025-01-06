import os
from huggingface_hub import login
from tqdm import tqdm
import pickle
login(os.getenv("HUGGINGFACE_TOKEN"))
from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoTokenizer, AutoProcessor,AutoModelForCausalLM
import numpy as np
from early_exit_model import EarlyExitWrapper
import argparse

def load_model():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer,model


def define_arguments(parser):
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index for early exit",
    )
    return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Early exit generation")
    parser = define_arguments(parser)
    args = parser.parse_args()
    
    tokenizer,model = load_model()
    question = "What is the capital of France?"
    layer_number = args.layer
    max_new_tokens = 20

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    input_tensor = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",return_dict=True 
    ).to(model.device)

    terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]


     # Wrap the model with early exit functionality
    wrapped_model = EarlyExitWrapper(model, layer_number)

    terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    # Generate tokens with the wrapped model
    generated_ids = wrapped_model.generate(
        input_ids = input_tensor["input_ids"],
        attention_mask = input_tensor["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id = terminators
    )

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0][input_tensor["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(generated_text)