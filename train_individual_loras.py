

from src.data.datasets import *
from src.utils import *
from src.model.lora  import *
from src.train.train_lora import *
from src.evaluation.task_evaluation import *

import argparse

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    task = args.task
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if task == "combined":
        train_dataset, val_dataset, train_loader, val_loader = get_combined_dataset(tokenizer)
        for param in model.parameters():
            param.requires_grad = False
        apply_lora_to_model(model, model_name="qwen")
        print_trainable_params(model)
        train_lora_with_val(model, train_loader, val_loader, epochs=10)

    else:
        train_data, val_data, train_dataset, val_dataset, train_loader, val_loader = get_datas_for_task(task, tokenizer, train_size=5000)
        evaluate_task(model, tokenizer, val_data, task=task)

        for param in model.parameters():
            param.requires_grad = False
        apply_lora_to_model(model, model_name="qwen")
        print_trainable_params(model)
        train_lora_with_val(model, train_loader, val_loader, epochs=10)

        evaluate_task(model, tokenizer, val_data, task=task)
        
    save_lora_weights(model, f"weights/{model_name.split('/')[-1]}/{task}_lora.pth")