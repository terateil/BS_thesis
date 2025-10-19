from torch.utils.data import Dataset
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

class PromptOnlyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, target_max_len=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for x in data:
            prompt = self.build_prompt(x)

            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )

            self.data.append({
                "prompt": encoded["input_ids"].squeeze(0),
                "prompt_mask": encoded["attention_mask"].squeeze(0)
            })

    def build_prompt(self, x):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SummarizationPromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        prompt = f"{x['article'].strip()}\n### Summarization:\n"
        return prompt

class TranslationPromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        prompt = f"{x['translation']['en'].strip()}\n### Translation from English to German:\n"
        return prompt

class QAPromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        context = x['context'].strip()
        question = x['question'].strip()
        prompt = f"### Context:\n{context}\n### Question:\n{question}\n### Answer:\n"
        return prompt

class SentimentPromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        prompt = f"{x['sentence'].strip() if 'sentence' in x else x['text'].strip()}\n### Sentiment:\n"
        return prompt

class NLIPromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        if x['label'] == -1:
            return "", ""
        premise = x['premise'].strip()
        hypothesis = x['hypothesis'].strip()
        prompt = f"### Premise:\n{premise}\n### Hypothesis:\n{hypothesis}\n### Relationship:\n"
        return prompt
    
class MathPromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        prompt = f"{x['question'].strip()}\n### Answer:\n"
        return prompt

class ParaphrasingPromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        prompt = f"{x['sentence1'].strip()}\n### Paraphrasing:\n"
        return prompt
    
class CommonsensePromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        goal = x['goal'].strip()
        sol1 = x['sol1'].strip()
        sol2 = x['sol2'].strip()
        prompt = f"{goal}\n### Solution 1:\n{sol1}\n### Solution 2:\n{sol2}\n### Correct Solution:\n"
        return prompt

class DialoguePromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        context = x['instruction']
        prompt = f"### Instruction:\n{context}\n### Response:\n"
        return prompt

class CodeCompletionPromptOnlyDataset(PromptOnlyDataset):
    def build_prompt(self, x):
        task_description = x['text'].strip()
        prompt = f"{task_description}\n### Code Solution:\n"
        return prompt



class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, target_max_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_max_len = target_max_len

        for x in data:
            prompt, target = self.build_prompt_and_target(x)

            prompt_max_len = self.max_len - self.target_max_len

            prompt_inputs = tokenizer(prompt, truncation=True, max_length=prompt_max_len, add_special_tokens=False)
            p_ids = prompt_inputs["input_ids"]
            t_ids = tokenizer(target, truncation=True, max_length=self.target_max_len - 1, add_special_tokens=False)["input_ids"]
            t_ids += [tokenizer.eos_token_id]

            input_ids = p_ids + t_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(p_ids) + t_ids

            pad_len = self.max_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len

            self.data.append({
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)
            })

    def build_prompt_and_target(self, x):
        """This method must be overridden for each task"""
        raise NotImplementedError

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class SummarizationDataset(CustomDataset):
    def build_prompt_and_target(self, x):
        prompt = f"{x['article'].strip()}\n### Summarization:\n"
        target = x['highlights'].strip()
        return prompt, target

class TranslationDataset(CustomDataset):
    def build_prompt_and_target(self, x):
        prompt = f"{x['translation']['en'].strip()}\n### Translation from English to German:\n"
        target = x['translation']['de'].strip()
        return prompt, target

class QADataset(CustomDataset):
    def build_prompt_and_target(self, x):
        context = x['context'].strip()
        question = x['question'].strip()
        answer = x['answers']['text'][0].strip()
        prompt = f"### Context:\n{context}\n### Question:\n{question}\n### Answer:\n"
        return prompt, answer

class SentimentDataset(CustomDataset):
    def build_prompt_and_target(self, x):
        label = "positive" if x['label'] == 1 else "negative"
        prompt = f"{x['sentence'].strip() if 'sentence' in x else x['text'].strip()}\n### Sentiment:\n"
        return prompt, label

class NLIDataset(CustomDataset):
    def build_prompt_and_target(self, x):
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        if x['label'] == -1:
            return "", ""
        premise = x['premise'].strip()
        hypothesis = x['hypothesis'].strip()
        label = label_map[x['label']]
        prompt = f"### Premise:\n{premise}\n### Hypothesis:\n{hypothesis}\n### Relationship:\n"
        return prompt, label
    
class MathDataset(CustomDataset):
    def build_prompt_and_target(self, x):
        prompt = f"{x['question'].strip()}\n### Answer:\n"
        target = x['answer'].strip()
        return prompt, target

class ParaphrasingDataset(CustomDataset):
    def build_prompt_and_target(self, x):
        prompt = f"{x['sentence1'].strip()}\n### Paraphrasing:\n"
        target = x['sentence2'].strip()
        return prompt, target
    
class CommonsenseDataset(CustomDataset):
    def build_prompt_and_target(self, x):
        goal = x['goal'].strip()
        sol1 = x['sol1'].strip()
        sol2 = x['sol2'].strip()
        label = x['label']
        prompt = f"{goal}\n### Solution 1:\n{sol1}\n### Solution 2:\n{sol2}\n### Correct Solution:\n"
        target = sol1 if label == 0 else sol2
        return prompt, target

class DialogueDataset(CustomDataset):
    def build_prompt_and_target(self, x):
        context = x['instruction']
        response = x['output']
        prompt = f"### Instruction:\n{context}\n### Response:\n"
        target = response
        return prompt, target

class CodeCompletionDataset(CustomDataset):
    def build_prompt_and_target(self, x):
        task_description = x['text'].strip()
        code_solution = x['code'].strip()
        prompt = f"{task_description}\n### Code Solution:\n"
        target = code_solution
        return prompt, target



def get_datas_for_task(
    task, tokenizer, batch_size=2, max_len=512, target_max_len=128,
    sample_ratio="1%", train_size=None, val_size=None, prompt_only=False
):
    # helper function to determine split string
    def get_split(base_split, size, ratio):
        if size is not None:
            return f"{base_split}[:{size}]"
        else:
            return f"{base_split}[:{ratio}]"

    if task == "summarization":
        train_split = get_split("train", train_size, sample_ratio)
        val_split = get_split("validation", val_size, sample_ratio)
        train_data = load_dataset("cnn_dailymail", "3.0.0", split=train_split)
        val_data = load_dataset("cnn_dailymail", "3.0.0", split=val_split)
        dataset_class = SummarizationDataset if not prompt_only else SummarizationPromptOnlyDataset

    elif task == "translation":
        train_split = get_split("train", train_size, sample_ratio)
        val_split = get_split("validation", val_size, sample_ratio)
        train_data = load_dataset("wmt14", "de-en", split=train_split)
        val_data = load_dataset("wmt14", "de-en", split=val_split)
        dataset_class = TranslationDataset if not prompt_only else TranslationPromptOnlyDataset

    elif task == "qa":
        train_split = get_split("train", train_size, sample_ratio)
        val_split = get_split("validation", val_size, sample_ratio)
        train_data = load_dataset("squad", split=train_split)
        val_data = load_dataset("squad", split=val_split)
        dataset_class = QADataset if not prompt_only else QAPromptOnlyDataset

    elif task == "sentiment":
        train_split = get_split("train", train_size, sample_ratio)
        val_split = get_split("validation", val_size, sample_ratio)
        train_data = load_dataset("glue", "sst2", split=train_split)
        val_data = load_dataset("glue", "sst2", split=val_split)
        dataset_class = SentimentDataset if not prompt_only else SentimentPromptOnlyDataset

    elif task == "nli":
        train_split = get_split("train", train_size, sample_ratio)
        val_split = get_split("validation", val_size, sample_ratio)
        train_data = load_dataset("snli", split=train_split).filter(lambda x: x['label'] != -1)
        val_data = load_dataset("snli", split=val_split).filter(lambda x: x['label'] != -1)
        dataset_class = NLIDataset if not prompt_only else NLIPromptOnlyDataset

    elif task == "math":
        train_split = get_split("train", train_size, sample_ratio)
        val_split = get_split("test", val_size, sample_ratio)  # GSM8K에는 validation 없음
        train_data = load_dataset("gsm8k", "main", split=train_split)
        val_data = load_dataset("gsm8k", "main", split=val_split)
        dataset_class = MathDataset if not prompt_only else MathPromptOnlyDataset

    elif task == "paraphrasing":
        train_split = get_split("train", train_size, sample_ratio)
        val_split = get_split("validation", val_size, sample_ratio)
        train_data = load_dataset("paws", "labeled_final", split=train_split)
        val_data = load_dataset("paws", "labeled_final", split=val_split)
        dataset_class = ParaphrasingDataset if not prompt_only else ParaphrasingPromptOnlyDataset

    elif task == "commonsense":
        train_split = get_split("train", train_size, sample_ratio)
        val_split = get_split("validation", val_size, sample_ratio)
        train_data = load_dataset("piqa", split=train_split)
        val_data = load_dataset("piqa", split=val_split)
        dataset_class = CommonsenseDataset if not prompt_only else CommonsensePromptOnlyDataset

    elif task == "dialogue":
        full_data = load_dataset("tatsu-lab/alpaca", split="train")
        sample_count = int(len(full_data) * float(sample_ratio.strip('%')) / 100)
        train_count = train_size if train_size is not None else sample_count
        val_count = val_size if val_size is not None else sample_count
        train_data = full_data.select(range(train_count))
        val_data = full_data.select(range(train_count, train_count + val_count))
        dataset_class = DialogueDataset if not prompt_only else DialoguePromptOnlyDataset
        
    elif task == "code":
        train_split = get_split("train", train_size, sample_ratio)
        val_split = get_split("test", val_size, sample_ratio)
        train_data = load_dataset("mbpp", split=train_split)
        val_data = load_dataset("mbpp", split=val_split)
        dataset_class = CodeCompletionDataset if not prompt_only else CodeCompletionPromptOnlyDataset


    else:
        raise ValueError(f"Unsupported task: {task}")

    # build dataset objects
    train_dataset = dataset_class(train_data, tokenizer, max_len=max_len, target_max_len=target_max_len)
    val_dataset = dataset_class(val_data, tokenizer, max_len=max_len, target_max_len=target_max_len)

    # return dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_data, val_data, train_dataset, val_dataset, train_loader, val_loader

############################################ Combined Dataset #########################################################

from torch.utils.data import ConcatDataset, DataLoader

def get_combined_dataset(tokenizer, tasks=["summarization", "translation", "qa", "sentiment", "nli"], batch_size=8, max_len=512, target_max_len=128, sample_ratio="1%"):
    all_train_datasets = []
    all_val_datasets = []

    for task in tasks:
        train_data, val_data, train_dataset, val_dataset, _, _ = get_datas_for_task(
            task, tokenizer, batch_size, max_len, target_max_len, train_size=5000
        )
        all_train_datasets.append(train_dataset)
        all_val_datasets.append(val_dataset)

    # Concatenate each
    combined_train_dataset = ConcatDataset(all_train_datasets)
    combined_val_dataset = ConcatDataset(all_val_datasets)

    # Shuffled DataLoader
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False)

    return combined_train_dataset, combined_val_dataset, train_loader, val_loader



################################################# Router Dataset #################################################


class MultiDomainRouterDataset(Dataset):
    def __init__(self, domain_datasets: dict):
        """
        domain_datasets: {domain_name: dataset_object}
        """
        self.samples = []
        self.domain2id = {name: i for i, name in enumerate(domain_datasets.keys())}

        for domain_name, dataset in domain_datasets.items():
            domain_id = self.domain2id[domain_name]
            for i in range(len(dataset)):
                item = dataset[i]
                item["domain_label"] = torch.tensor(domain_id)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def get_router_dataloader(tokenizer, task_names=["summarization", "translation", "qa", "sentiment", "nli"], batch_size=4, sample_ratio="1%", max_len=512, target_max_len=128):
    domain_datasets = {}

    for task in task_names:
        _, _, train_dataset, _, _, _ = get_datas_for_task(
            task, tokenizer, batch_size=batch_size,
            max_len=max_len, target_max_len=target_max_len,
            sample_ratio=sample_ratio
        )
        domain_datasets[task] = train_dataset  # Put as is

    combined_dataset = MultiDomainRouterDataset(domain_datasets)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True), domain_datasets

def get_router_dataloader_batched(
    tokenizer,
    task_names=["summarization", "translation", "qa", "sentiment", "nli"],
    batch_size=4,
    train_sample_ratio="1%",
    val_sample_ratio="0.5%",
    max_len=512,
    target_max_len=128
):
    domain2id = {name: i for i, name in enumerate(task_names)}
    train_batches = []
    val_batches = []

    task_train_datasets = {}
    task_val_datasets = {}

    # 1. Load all datasets first
    for task in task_names:
        _, _, train_dataset, val_dataset, _, _ = get_datas_for_task(
            task,
            tokenizer,
            batch_size=batch_size,
            max_len=max_len,
            target_max_len=target_max_len,
            train_size=5000,
            val_size=100,
            prompt_only=True
        )
        task_train_datasets[task] = train_dataset
        task_val_datasets[task] = val_dataset

    # 2. Find min dataset length across tasks
    min_train_len = min(len(ds) for ds in task_train_datasets.values())
    min_val_len = min(len(ds) for ds in task_val_datasets.values())

    print(f"Min train length: {min_train_len}")
    print(f"Min val length: {min_val_len}")

    # 3. Truncate and batch
    for task in task_names:
        domain_id = domain2id[task]

        # Train
        train_dataset = task_train_datasets[task]
        train_subset = torch.utils.data.Subset(train_dataset, range(min_train_len))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        for batch in train_loader:
            batch["domain_label"] = torch.full((batch["prompt"].size(0),), domain_id)
            train_batches.append(batch)

        # Val
        val_dataset = task_val_datasets[task]
        val_subset = torch.utils.data.Subset(val_dataset, range(min_val_len))
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        for batch in val_loader:
            batch["domain_label"] = torch.full((batch["prompt"].size(0),), domain_id)
            val_batches.append(batch)

    # Shuffle train batches globally
    import random
    random.shuffle(train_batches)

    return train_batches, val_batches

