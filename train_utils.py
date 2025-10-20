import os
import torch
import torch.nn as nn

def seed_torch(seed: int, deterministic: str = "off") -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): The seed value.
        deterministic (str): One of {"off", "seed_only", "strict"}.
    """
    if deterministic not in {"off", "seed_only", "strict"}:
        raise ValueError(f"Invalid deterministic mode: {deterministic}")

    if deterministic == "off":
        return

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic == "strict":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # seed_only
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def validate_gpu_ids(gpu_ids: str) -> list:
    """Validate GPU IDs and return list of available GPUs"""
    indices = [int(idx) for idx in gpu_ids.split(',')]
    available = torch.cuda.device_count()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
        
    invalid = [idx for idx in indices if idx >= available]
    if invalid:
        raise ValueError(f"GPU indices {invalid} are not available. Available GPUs: 0-{available-1}")
    
    return indices

def initialize_weights(module):
    """Initialize model weights"""
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.constant_(module.in_proj_bias, 0)


def get_optimizer(model, params):
    initial_lr = params['lr']
    if params['adjust_lr_with_batch_size']:
        batch_size = params["batch_size"]
        gradient_accumulation_steps = params["gradient_accumulation_steps"]
        initial_lr = initial_lr * batch_size * gradient_accumulation_steps
    weight_decay = params.get('weight_decay', 1e-5)

    return torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
