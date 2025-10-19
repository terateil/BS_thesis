import os

import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from contextlib import contextmanager

##############################################################################################
## LoRA
##############################################################################################

class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        in_f, out_f = linear.in_features, linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias)
        lora = self.dropout(x) @ self.A.T @ self.B.T * self.scaling
        return base + lora
    

    
def get_parent_module(model, name):
    parts = name.split('.')
    for p in parts[:-1]:
        model = getattr(model, p)
    return model




def apply_lora(model, target_keywords, r=8, alpha=16, dropout=0.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            parent = get_parent_module(model, name)
            child = name.split('.')[-1]
            setattr(parent, child, LoRALinear(module, r, alpha, dropout))


def apply_lora_to_model(model, model_name=None, target_keywords=None,
                        r=8, alpha=16, dropout=0.0, lora_ckpt_path=None):

    if target_keywords is None:
        if model_name is None:
            raise ValueError("`model_name` or `target_keywords` must be provided.")

        if "qwen" in model_name.lower():
            target_keywords = [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
            ]
        # elif "llama" in model_name.lower():
        #     target_keywords = [
        #         "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        #         "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"
        #     ]
        # elif "gpt2" in model_name.lower():
        #     target_keywords = [
        #         "attn.c_attn", "mlp.c_fc", "mlp.c_proj"
        #     ]
        else:
            raise ValueError(f"Unknown model_name: {model_name}. Please specify target_keywords manually.")

    # Apply LoRA
    apply_lora(model, target_keywords, r, alpha, dropout)

    # Load checkpoint
    if lora_ckpt_path and os.path.exists(lora_ckpt_path):
        print(f"Loading LoRA weights from {lora_ckpt_path}")
        lora_state_dict = torch.load(lora_ckpt_path, map_location="cpu")
        model.load_state_dict(lora_state_dict, strict=False)
    else:
        print("Initialized new LoRA layers.")


def remove_lora_from_model(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            parent = get_parent_module(model, name)
            child_name = name.split('.')[-1]
            restored = nn.Linear(module.A.shape[1], module.B.shape[0])
            restored.weight.data = module.weight.data.clone()
            if module.bias is not None:
                restored.bias = nn.Parameter(module.bias.data.clone())
            setattr(parent, child_name, restored)
    print("All LoRA layers removed and restored to original Linear.")

def save_lora_weights(model, save_path):
    lora_state_dict = {
        k: v for k, v in model.state_dict().items()
        if 'A' in k or 'B' in k  # LoRALinear 내부 파라미터
    }
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(lora_state_dict, save_path)
    print(f"Saved LoRA weights to {save_path}")

##############################################################################################
## MoA
##############################################################################################

SHARED_ROUTER_MODE = False
SHARED_ROUTER_DECISION = None

@contextmanager
def shared_router(expert_id=None):
    """
    Context manager to enable shared routing.
    If expert_id is given, force all layers to use that expert.
    Otherwise, use the first router's choice.
    """
    global SHARED_ROUTER_MODE, SHARED_ROUTER_DECISION
    old_mode = SHARED_ROUTER_MODE
    old_decision = SHARED_ROUTER_DECISION

    SHARED_ROUTER_MODE = True
    SHARED_ROUTER_DECISION = expert_id  # None means the first router will decide

    try:
        yield
    finally:
        # Restore state
        SHARED_ROUTER_MODE = old_mode
        SHARED_ROUTER_DECISION = old_decision


class MultiLoRALinearWithRouter(nn.Module):
    def __init__(self, linear: nn.Linear, expert_names, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        in_f, out_f = linear.in_features, linear.out_features
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        self.expert_names = expert_names
        self.experts = nn.ModuleDict({
            name: nn.ParameterDict({
                "A": nn.Parameter(torch.randn(r, in_f) * 0.01),
                "B": nn.Parameter(torch.zeros(out_f, r))
            }) for name in expert_names
        })

        self.router = nn.Linear(in_f, len(expert_names))

        # Internal state
        self._active_expert_id = None  # For forced expert selection
        self._router_label = None      # For training labels
        self._router_loss = None       # Collected from external training loop
        self.last_logits = None        # For accuracy evaluation

    def set_active_expert(self, expert_id: int):
        self._active_expert_id = expert_id

    def clear_active_expert(self):
        self._active_expert_id = None

    def set_router_label(self, label):
        self._router_label = label

    def clear_router_label(self):
        self._router_label = None
        self._router_loss = None
        self.last_logits = None

    def get_router_loss(self):
        return self._router_loss

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        pooled = x[:, -1, :]

        # Router training
        if self._router_label is not None:
            logits = self.router(pooled)
            self.last_logits = logits.detach()
            self._router_loss = F.cross_entropy(logits, self._router_label)

        global SHARED_ROUTER_MODE, SHARED_ROUTER_DECISION

        # shared router mode
        if SHARED_ROUTER_MODE:
            if SHARED_ROUTER_DECISION is None:
                logits = self.router(pooled)
                SHARED_ROUTER_DECISION = logits.argmax(dim=-1)[0].item()
            expert_id = SHARED_ROUTER_DECISION
            

        elif self._active_expert_id is not None:
            expert_id = self._active_expert_id

        else:
            logits = self.router(pooled)
            expert_id = logits.argmax(dim=-1)[0].item()

        expert_name = self.expert_names[expert_id]
        A = self.experts[expert_name]["A"]
        B = self.experts[expert_name]["B"]
        lora = self.dropout(x) @ A.T @ B.T * self.scaling
        return base + lora





def apply_moa_lora(model, target_keywords, expert_names, r=8, alpha=16, dropout=0.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            parent = get_parent_module(model, name)
            child = name.split('.')[-1]
            moa_wrapper = MultiLoRALinearWithRouter(module, expert_names, r, alpha, dropout)
            setattr(parent, child, moa_wrapper)
    print(f"Applied MoA with {len(expert_names)} experts.")


def apply_moa_to_model(model, model_name=None, target_keywords=None,
                       expert_names=["summarization", "translation", "qa", "sentiment", "nli"],
                       r=8, alpha=16, dropout=0.0,
                       lora_ckpt_dir=None):
    """
    Apply MoA structure to the model.
    """
    if not expert_names:
        raise ValueError("expert_names must be specified.")

    if target_keywords is None:
        if model_name is None:
            raise ValueError("`model_name` or `target_keywords` must be provided.")

        if "qwen" in model_name.lower():
            target_keywords = [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
            ]
        # elif "llama" in model_name.lower():
        #     target_keywords = [
        #         "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        #         "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"
        #     ]
        # elif "gpt2" in model_name.lower():
        #     target_keywords = [
        #         "attn.c_attn", "mlp.c_fc", "mlp.c_proj"
        #     ]
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    # Step 1: Apply MoA structure
    apply_moa_lora(model, target_keywords, expert_names, r, alpha, dropout)
    print(f"MoA structure applied. Expert list: {expert_names}")

    # Step 2: Load LoRA checkpoint for each expert
    if lora_ckpt_dir and os.path.isdir(lora_ckpt_dir):
        for expert_name in expert_names:
            ckpt_path = os.path.join(lora_ckpt_dir, f"{expert_name}_lora.pth")
            if not os.path.exists(ckpt_path):
                print(f"LoRA checkpoint not found for expert '{expert_name}': {ckpt_path}")
                continue

            print(f"Loading LoRA weights for expert '{expert_name}' from {ckpt_path}")
            lora_state = torch.load(ckpt_path, map_location="cpu")

            # Copy to MoA structure inside
            for name, module in model.named_modules():
                if isinstance(module, MultiLoRALinearWithRouter):
                    prefix = f"{name}."
                    A_key = f"{prefix}A"
                    B_key = f"{prefix}B"

                    if A_key in lora_state and B_key in lora_state:
                        with torch.no_grad():
                            module.experts[expert_name]["A"].copy_(lora_state[A_key])
                            module.experts[expert_name]["B"].copy_(lora_state[B_key])
                    else:
                        print(f"Skipped expert '{expert_name}' for layer '{name}': A/B not found in checkpoint")
    else:
        print("MoA experts initialized from scratch (no checkpoints provided)")

def save_moa_weights(model, save_path):
    """
    Save all trainable parameters in MultiLoRALinearWithRouter:
    - Each expert's A, B
    - Router's classifier
    """
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinearWithRouter):
            for expert_name, params in module.experts.items():
                state_dict[f"{name}.experts.{expert_name}.A"] = params["A"]
                state_dict[f"{name}.experts.{expert_name}.B"] = params["B"]
            state_dict[f"{name}.router.weight"] = module.router.weight
            if module.router.bias is not None:
                state_dict[f"{name}.router.bias"] = module.router.bias

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state_dict, save_path)
    print(f"Saved MOA weights to: {save_path}")

def load_moa_weights(model, load_path, strict=False):
    """
    Load saved MOA state_dict into MultiLoRALinearWithRouter.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    print(f"Loading MOA weights from: {load_path}")
    loaded_state = torch.load(load_path, map_location="cpu")

    # Saved key structure: "layer_name.experts.expert_name.A"
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinearWithRouter):
            for expert_name in module.expert_names:
                A_key = f"{name}.experts.{expert_name}.A"
                B_key = f"{name}.experts.{expert_name}.B"
                if A_key in loaded_state:
                    module.experts[expert_name]["A"].data.copy_(loaded_state[A_key])
                if B_key in loaded_state:
                    module.experts[expert_name]["B"].data.copy_(loaded_state[B_key])

            # router도 복원
            r_w = f"{name}.router.weight"
            r_b = f"{name}.router.bias"
            if r_w in loaded_state:
                module.router.weight.data.copy_(loaded_state[r_w])
            if r_b in loaded_state and module.router.bias is not None:
                module.router.bias.data.copy_(loaded_state[r_b])

    print("Loaded MOA weights.")


def set_moa_trainable(model, router=True, lora=True):
    """
    Set the trainability of the Router and each LoRA expert in the model.
    """
    count_total = 0
    count_enabled = 0

    for module in model.modules():
        if isinstance(module, MultiLoRALinearWithRouter):
            # Router
            for p in module.router.parameters():
                p.requires_grad = router
                count_total += p.numel()
                if router:
                    count_enabled += p.numel()

            # Each expert's A, B
            for expert in module.experts.values():
                for p in expert.values():
                    p.requires_grad = lora
                    count_total += p.numel()
                    if lora:
                        count_enabled += p.numel()

    print(f"MOA trainable params: {count_enabled:,} / {count_total:,} ({(count_enabled / count_total * 100):.2f}%)")
    




################################# Merging Strategies #################################

class ConcatLoRALinear(nn.Module):
    def __init__(self, linear, lora_list, alpha=16, r=8):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # Multiple LoRA modules
        self.loras = nn.ModuleList()
        for state in lora_list:
            A = nn.Parameter(state["A"].clone())
            B = nn.Parameter(state["B"].clone())
            scaling = alpha / r
            self.loras.append(nn.ParameterDict({"A": A, "B": B, "scaling": torch.tensor(scaling)}))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora_total = 0
        for expert in self.loras:
            A, B, scale = expert["A"], expert["B"], expert["scaling"]
            lora = x @ A.T @ B.T * scale
            # lora = x @ A.T @ B.T
            lora_total += lora
        return base + lora_total

def merge_lora_weights(lora_dicts, weights):
    merged = {}
    for key in lora_dicts[0].keys():
        merged[key] = sum(w * l[key] for w, l in zip(weights, lora_dicts))
    return merged


def svd_merge_loras(lora_dicts, rank=8):
    merged = {}
    keys = [k for k in lora_dicts[0] if "A" in k]

    for k in keys:
        A_stack = torch.stack([l[k] for l in lora_dicts])  # [N, r, in]
        B_stack = torch.stack([l[k.replace("A", "B")] for l in lora_dicts])  # [N, out, r]

        AB = torch.einsum("nri,nor->noi", A_stack, B_stack)  # [N, out, in]
        AB_mean = AB.mean(dim=0)

        U, S, Vh = torch.linalg.svd(AB_mean, full_matrices=False)
        A_new = (S[:rank].unsqueeze(1) * Vh[:rank, :]).contiguous()
        B_new = U[:, :rank].contiguous()

        merged[k] = A_new
        merged[k.replace("A", "B")] = B_new

    return merged

def apply_lora_merged(model, model_name, lora_dicts, merge_type="weighted", weights=None,
                       r=8, alpha=16, target_keywords=None):
    if target_keywords is None:
        if "qwen" in model_name.lower():
            target_keywords = [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
            ]
        # elif "llama" in model_name.lower():
        #     target_keywords = [
        #         "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        #         "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"
        #     ]
        # elif "gpt2" in model_name.lower():
        #     target_keywords = [
        #         "attn.c_attn", "mlp.c_fc", "mlp.c_proj"
        #     ]
        else:
            raise ValueError("Unknown model type")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            parent = get_parent_module(model, name)
            child = name.split('.')[-1]

            if merge_type == "concat":
                # assume all lora_dicts have keys like 'layername.A', 'layername.B'
                lora_states = []
                for state in lora_dicts:
                    A = state[name + ".A"]
                    B = state[name + ".B"]
                    lora_states.append({"A": A, "B": B})
                new_module = ConcatLoRALinear(linear=module, lora_list=lora_states, alpha=alpha, r=r)

            elif merge_type == "weighted":
                # weights = [1/len(lora_dicts) for _ in lora_dicts]
                weights = [1 for _ in lora_dicts]
                merged_state = merge_lora_weights([{
                    name + ".A": l[name + ".A"],
                    name + ".B": l[name + ".B"]
                } for l in lora_dicts], weights)
                new_module = LoRALinear(module, r, alpha)
                new_module.A.data = merged_state[name + ".A"]
                new_module.B.data = merged_state[name + ".B"]

            elif merge_type == "svd":
                merged_state = svd_merge_loras([{
                    name + ".A": l[name + ".A"],
                    name + ".B": l[name + ".B"]
                } for l in lora_dicts], rank=r)
                new_module = LoRALinear(module, r, alpha)
                new_module.A.data = merged_state[name + ".A"]
                new_module.B.data = merged_state[name + ".B"]

            else:
                raise ValueError("Unknown merge_type")

            setattr(parent, child, new_module)

    print(f"Applied LoRA using merge_type = {merge_type}")
