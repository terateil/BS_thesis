

from src.data.datasets import *
from src.utils import *
from src.model.lora  import *
from src.model.lora import SHARED_ROUTER_MODE, SHARED_ROUTER_DECISION
from src.train.train_lora import *
from src.evaluation.task_evaluation import *

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--model_type", type=str, default="moa")  # base | single | moa | concat | weighted | svd
    args = parser.parse_args()

    model_name = args.model_name
    model_type = args.model_type
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    tasks = ["summarization", "translation", "qa", "sentiment", "nli"]
    lora_paths = [f"weights/{model_name.split('/')[-1]}/{task}_lora.pth" for task in tasks]
    os.makedirs(f"results/{model_name.split('/')[-1]}/{len(tasks)}", exist_ok=True)
    log_file = open(f"results/{model_name.split('/')[-1]}/{len(tasks)}/{model_type}.txt", "w")

    if model_type == "base":
        print("Using base model without any LoRA or MoA.")

    if model_type == "combined":
        print("Applying combined model.")
        apply_lora_to_model(model, model_name="qwen", lora_ckpt_path=f"weights/{model_name.split('/')[-1]}/combined_lora.pth")

    elif model_type == "moa":
        print("Applying MoA structure.")
        apply_moa_to_model(model, model_name="qwen", expert_names=tasks)
        load_moa_weights(model, f"weights/{model_name.split('/')[-1]}/MoA/moa_weights_stable_{len(tasks)}.pth")

    elif model_type in ["concat", "weighted", "svd"]:
        print(f"Merging LoRAs with method: {model_type}")
        lora_dicts = [torch.load(p, map_location="cpu") for p in lora_paths]
        apply_lora_merged(
            model,
            model_name="qwen",
            lora_dicts=lora_dicts,
            merge_type=model_type,
        )


    for i,task in enumerate(tasks):
        if model_type == "single":
            print("Applying single LoRA.")
            apply_lora_to_model(
                model,
                model_name="qwen",
                lora_ckpt_path=lora_paths[i]
            )
            
        _, val_data, _, _, _, _ = get_datas_for_task(task, tokenizer, batch_size=1 if model_type == "moa" else 16, val_size=500)
        evaluate_task(model, tokenizer, val_data, task=task, batch_size=1 if model_type == "moa" else 16, log_file=log_file)
        
        if model_type == "single":
            remove_lora_from_model(model)

    log_file.close()

    print(f"Results saved to {log_file.name}")