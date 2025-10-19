from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW

def evaluate_val_loss(model, val_loader, device="cuda"):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            x = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=x['input_ids'], attention_mask=x['attention_mask']).logits

            #  Shift for next-token prediction
            shift_logits = out[:, :-1, :].contiguous()
            shift_labels = x['labels'][:, 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))

            num_tokens = shift_labels.ne(-100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")

def train_lora_with_val(
    model,
    train_loader,
    val_loader,
    lr=5e-5,
    epochs=3,
    val_every=100,
    patience=3,
    device="cuda"
):
    model.to(device)
    model.train()
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    steps_since_improvement = 0
    step = 0

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            x = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits

            shift_logits = out[:, :-1, :].contiguous()
            shift_labels = x["labels"][:, 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(train_loss=f"{loss.item():.4f}")
            step += 1

            if step % val_every == 0:
                val_loss = evaluate_val_loss(model, val_loader, device=device)
                print(f"Step {step}: Validation loss = {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    steps_since_improvement = 0
                    print("Validation loss improved.")
                else:
                    steps_since_improvement += 1
                    print(f"No improvement. Patience: {steps_since_improvement}/{patience}")
                    if steps_since_improvement >= patience:
                        print("Early stopping triggered.")
                        return
