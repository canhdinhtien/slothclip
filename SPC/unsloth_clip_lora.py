import torch
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd

torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}

class FastCLIP_LoRA_Function(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, X: torch.Tensor, W, A, B, scaling):
        orig_shape = X.shape
        if X.dim() == 3:
            X = X.view(-1, X.shape[-1])
            
        XW = F.linear(X, W)
        X_lora = F.linear(F.linear(X, A), B) * scaling
        
        out = XW + X_lora
        
        ctx.save_for_backward(X, W, A, B)
        ctx.scaling = scaling
        ctx.orig_shape = orig_shape
        
        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)
            
        return out

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        X, W, A, B = ctx.saved_tensors
        scaling = ctx.scaling
        
        target_dtype = grad_output.dtype
        
        X = X.to(target_dtype).reshape(-1, X.shape[-1])
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        W = W.to(target_dtype)
        A = A.to(target_dtype)
        B = B.to(target_dtype)
        
        d_A_t = (X.t() @ (grad_output @ B)) * scaling
        d_B_t = ((A @ X.t()) @ grad_output) * scaling

        dX = (grad_output @ W) + ((grad_output @ B) @ A) * scaling
        
        if len(ctx.orig_shape) == 3:
            dX = dX.view(ctx.orig_shape[0], ctx.orig_shape[1], -1)
            
        return dX, None, d_A_t.t(), d_B_t.t(), None

class FastUnslothLoRALinear(torch.nn.Module):
    """Module LoRA thay thế nn.Linear, chạy trên Custom Autograd của Unsloth"""
    def __init__(self, base_layer, r=16, alpha=32, dropout=0.1):
        super().__init__()
        self.base_layer = base_layer # nn.Linear (bị freeze)
        self.r = r
        self.scaling = alpha / r
        
        self.lora_A = torch.nn.Parameter(torch.zeros((r, base_layer.in_features)))
        self.lora_B = torch.nn.Parameter(torch.zeros((base_layer.out_features, r)))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        
        self.dropout = torch.nn.Dropout(p=dropout)
        
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    @property
    def weight(self):
        return self.base_layer.weight
        
    @property
    def bias(self):
        return self.base_layer.bias

    @property
    def in_features(self):
        return self.base_layer.in_features
        
    @property
    def out_features(self):
        return self.base_layer.out_features

    def forward(self, x):
        x_dropped = self.dropout(x)
        out = FastCLIP_LoRA_Function.apply(
            x_dropped, 
            self.base_layer.weight, 
            self.lora_A, 
            self.lora_B, 
            self.scaling
        )
        if self.base_layer.bias is not None:
            out = out + self.base_layer.bias
        return out