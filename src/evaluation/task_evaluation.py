from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from ..model.lora import MultiLoRALinearWithRouter, shared_router
import sys
from evaluate import load

def evaluate_task(
    model,
    tokenizer,
    dataset,
    task="summarization",
    batch_size=4,
    num_samples=32,
    max_input_len=400,
    max_gen_len=128,
    device="cuda",
    log_file=sys.stdout
):
    model.eval()
    model.to(device)

    prompts, targets = [], []

    # for sample in dataset.select(range(min(num_samples, len(dataset)))):
    for sample in dataset:
        if task == "summarization":
            prompt = f"{sample['article'].strip()}\n### Summarization:\n"
            target = sample['highlights'].strip()

        elif task == "translation":
            prompt = f"{sample['translation']['en'].strip()}\n### Translation from English to German:\n"
            target = sample['translation']['de'].strip()

        elif task == "qa":
            context = sample['context'].strip()
            question = sample['question'].strip()
            answer = sample['answers']['text'][0].strip()
            prompt = f"### Context:\n{context}\n### Question:\n{question}\n### Answer:\n"
            target = answer

        elif task == "sentiment":
            text = sample.get('text', sample.get('sentence', '')).strip()
            label = "positive" if sample['label'] == 1 else "negative"
            prompt = f"{text}\n### Sentiment:\n"
            target = label

        elif task == "nli":
            if sample['label'] == -1:
                continue
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
            premise = sample['premise'].strip()
            hypothesis = sample['hypothesis'].strip()
            prompt = f"### Premise:\n{premise}\n### Hypothesis:\n{hypothesis}\n### Relationship:\n"
            target = label_map[sample['label']]

        elif task == "math":
            prompt = f"{sample['question'].strip()}\n### Answer:\n"
            target = sample['answer'].strip()

        elif task == "paraphrasing":
            prompt = f"{sample['sentence1'].strip()}\n### Paraphrasing:\n"
            target = sample['sentence2'].strip()

        elif task == "commonsense":
            goal = sample['goal'].strip()
            sol1 = sample['sol1'].strip()
            sol2 = sample['sol2'].strip()
            label = sample['label']
            prompt = f"{goal}\n### Solution 1:\n{sol1}\n### Solution 2:\n{sol2}\n### Correct Solution:\n"
            target = sol1 if label == 0 else sol2

        elif task == "dialogue":
            context = sample['instruction']
            response = sample['output']
            prompt = f"### Instruction:\n{context}\n### Response:\n"
            target = response

        elif task == "code":
            task_description = sample['text'].strip()
            code_solution = sample['code'].strip()
            prompt = f"{task_description}\n### Code Solution:\n"
            target = code_solution

        else:
            raise ValueError(f"Unsupported task: {task}")

        prompts.append(prompt)
        targets.append(target)

    encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=max_input_len)
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    dataloader = DataLoader(list(zip(input_ids, attention_mask, prompts, targets)),
                            batch_size=batch_size)

    all_preds = []
    all_refs = []

    is_moa_model = any(isinstance(m, MultiLoRALinearWithRouter) for m in model.modules())

    for batch in tqdm(dataloader, desc=f"Evaluating [{task}]"):
        x_ids, x_mask, x_prompts, x_targets = batch
        x_ids = x_ids.to(device)
        x_mask = x_mask.to(device)

        if is_moa_model:
            for m in model.modules():
                if isinstance(m, MultiLoRALinearWithRouter):
                    m.clear_router_label()
                    m.clear_active_expert()


        with torch.no_grad():
            # with shared_router():
            outputs = model.generate(
                input_ids=x_ids,
                attention_mask=x_mask,
                max_new_tokens=max_gen_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
        )

            

        for out, inp, prompt, ref in zip(outputs, x_ids, x_prompts, x_targets):
            gen_only = out[len(inp):]
            gen_text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
            all_preds.append(gen_text)
            all_refs.append(ref.strip())


    # ----------------------------
    # Task-specific metric
    # ----------------------------
    if task == "summarization":
        
        rouge = load("rouge")
        scores = rouge.compute(predictions=all_preds, references=all_refs)
        print(f"{task}\n📈 ROUGE:", scores, file=log_file, flush=True)
        return scores

    elif task == "translation":
        
        bleu = load("bleu")
        scores = bleu.compute(predictions=all_preds, references=all_refs)
        print(f"{task}\n📘 BLEU:", scores, file=log_file, flush=True)
        return scores

    elif task == "qa":
        
        squad = load("squad")

        predictions = []
        references = []

        for idx, (pred, ref) in enumerate(zip(all_preds, all_refs)):
            qid = str(idx)
            predictions.append({"id": qid, "prediction_text": pred})
            references.append({
                "id": qid,
                "answers": {
                    "text": [ref],
                    "answer_start": [0]  # dummy start to satisfy SQuAD format
                }
            })

        scores = squad.compute(predictions=predictions, references=references)
        print(f"{task}\nQA Metrics (EM/F1):", scores, file=log_file, flush=True)
        return scores



    elif task in ["sentiment", "nli"]:
        correct = sum([pred.lower() == ref.lower() for pred, ref in zip(all_preds, all_refs)])
        acc = correct / len(all_preds)
        print(f"{task}\n✅ Accuracy: {acc:.4f}", file=log_file, flush=True)
        return {"accuracy": acc}
    
    elif task == "math":
        def extract_number(s):
            import re
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            return numbers[0] if numbers else ""

        correct = sum([
            extract_number(pred) == extract_number(ref)
            for pred, ref in zip(all_preds, all_refs)
        ])
        acc = correct / len(all_preds)
        print(f"{task}\n🧮 Numeric Match Accuracy: {acc:.4f}", file=log_file, flush=True)
        return {"accuracy": acc}
    
    elif task == "paraphrasing":
        bleu = load("bleu")
        scores = bleu.compute(predictions=all_preds, references=all_refs)
        print(f"{task}\n📝 BLEU (paraphrase quality):", scores, file=log_file, flush=True)
        return scores

    elif task == "commonsense":
        correct = sum([pred.strip().lower() == ref.strip().lower() for pred, ref in zip(all_preds, all_refs)])
        acc = correct / len(all_preds)
        print(f"{task}\n🧠 Commonsense Accuracy: {acc:.4f}", file=log_file, flush=True)
        return {"accuracy": acc}
    
    elif task == "dialogue":
        # Reference-free metric like BLEU or PPL can be used. BLEU here for simplicity
        bleu = load("bleu")
        scores = bleu.compute(predictions=all_preds, references=all_refs)
        print(f"{task}\n💬 Dialogue BLEU:", scores, file=log_file, flush=True)
        return scores

    elif task == "code":
        # Exact match is too harsh, use BLEU for now
        bleu = load("bleu")
        scores = bleu.compute(predictions=all_preds, references=all_refs)
        print(f"{task}\n🧑‍💻 Code Completion BLEU:", scores, file=log_file, flush=True)
        return scores


    else:
        print("No metrics computed.")
        return all_preds, all_refs
