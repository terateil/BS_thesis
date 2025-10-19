from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from ..model.lora import MultiLoRALinearWithRouter

def evaluate_router_val_loss(model, val_batches, device="cuda"):
    model.eval()
    total_loss, total_tokens = 0, 0
    total_correct, total_preds = 0, 0

    with torch.no_grad():
        for batch in val_batches:
            input_ids = batch["prompt"].to(device)
            attention_mask = batch["prompt_mask"].to(device)
            domain_label = batch["domain_label"].to(device)

            for m in model.modules():
                if isinstance(m, MultiLoRALinearWithRouter):
                    m.set_active_expert(domain_label[0].item())
                    m.set_router_label(domain_label)

            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            batch_loss = 0
            count = 0
            for m in model.modules():
                if isinstance(m, MultiLoRALinearWithRouter):
                    loss = m.get_router_loss()
                    if loss is not None:
                        batch_loss += loss.item()
                        count += 1
                    
                    if m.last_logits is not None:
                        preds = torch.argmax(m.last_logits, dim=-1)
                        correct = (preds == domain_label).sum().item()
                        total_correct += correct
                        total_preds += domain_label.size(0)

            if count > 0:
                total_loss += batch_loss / count
                total_tokens += 1

            for m in model.modules():
                if isinstance(m, MultiLoRALinearWithRouter):
                    m.clear_active_expert()
                    m.clear_router_label()

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    accuracy = total_correct / total_preds if total_preds > 0 else 0.0
    return avg_loss, accuracy


def train_all_routers(
    model,
    train_batches,
    val_batches=None,
    lr=1e-4,
    epochs=1,
    val_every=100,
    patience=3,
    device="cuda"
):
    model.to(device)
    model.train()

    router_params = [
        p for m in model.modules()
        if isinstance(m, MultiLoRALinearWithRouter)
        for p in m.router.parameters()
    ]
    opt = AdamW(router_params, lr=lr)

    best_val_loss = float("inf")
    steps_since_improvement = 0
    step = 0

    for epoch in range(epochs):
        pbar = tqdm(train_batches, desc=f"[Router Training - Epoch {epoch+1}]")
        total_loss, total_acc, total_samples = 0, 0, 0

        for batch in pbar:
            input_ids = batch["prompt"].to(device)
            attention_mask = batch["prompt_mask"].to(device)
            domain_label = batch["domain_label"].to(device)

            # Fix domain
            for m in model.modules():
                if isinstance(m, MultiLoRALinearWithRouter):
                    assert (domain_label == domain_label[0]).all()
                    m.set_active_expert(domain_label[0].item())
                    m.set_router_label(domain_label)

            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            losses, accs = [], []
            for m in model.modules():
                if isinstance(m, MultiLoRALinearWithRouter):
                    if m.get_router_loss() is not None:
                        losses.append(m.get_router_loss())
                    if m.last_logits is not None:
                        preds = m.last_logits.argmax(dim=-1)
                        acc = (preds == domain_label).float().mean().item()
                        accs.append(acc)

            if not losses:
                continue

            loss = sum(losses) / len(losses)
            avg_acc = 100 * sum(accs) / len(accs)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_acc += avg_acc * batch_size
            total_samples += batch_size
            step += 1

            avg_loss_so_far = total_loss / total_samples
            avg_acc_so_far = total_acc / total_samples
            pbar.set_postfix(
                loss=f"{avg_loss_so_far:.4f}",
                acc=f"{avg_acc_so_far:.2f}%"
            )

            for m in model.modules():
                if isinstance(m, MultiLoRALinearWithRouter):
                    m.clear_active_expert()
                    m.clear_router_label()

            #validation check
            if val_batches and step % val_every == 0:
                val_loss, val_acc = evaluate_router_val_loss(model, val_batches, device)
                print(f"🔍 Step {step}: Validation loss = {val_loss:.4f}, Validation accuracy = {val_acc:.2f}%")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    steps_since_improvement = 0
                    print(" Validation improved")
                else:
                    steps_since_improvement += 1
                    print(f" No improvement. Patience {steps_since_improvement}/{patience}")
                    if steps_since_improvement >= patience:
                        print(" Early stopping triggered.")
                        return






def get_all_routers(model):
    """
    Return a list of all the routers in the model.
    """
    routers = []
    for module in model.modules():
        if isinstance(module, MultiLoRALinearWithRouter):
            routers.append(module.router)
    return routers


router_losses = []
