

from src.data.datasets import *
from src.model.lora import *
from src.train.train_lora import *
from src.train.train_moa import *
from src.evaluation.task_evaluation import *
from src.utils import *

from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    for param in model.parameters():
        param.requires_grad = False

    task_names = ["summarization", "translation", "qa", "sentiment", "nli"]

    apply_moa_to_model(model, model_name="qwen", 
                       expert_names = task_names, 
                       lora_ckpt_dir=os.path.abspath(f"weights/{model_name.split('/')[-1]}/"))

    set_moa_trainable(model, router=True, lora=False)
    print_trainable_params(model)
    train_batches, val_batches = get_router_dataloader_batched(tokenizer, task_names=task_names)
    train_all_routers(model, train_batches)

    save_moa_weights(model, f"weights/{model_name.split('/')[-1]}/MoA/moa_weights_stable_{len(task_names)}.pth")