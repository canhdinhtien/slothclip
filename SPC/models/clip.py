"""
Code mainly from https://github.com/vturrisi/disef/blob/main/fine-tune/src/model.py
"""

import types
import torch
import torch.nn as nn
import clip
from torch.utils.checkpoint import checkpoint

import torch.nn.functional as F

from subset_names import SUBSET_NAMES
from templates import TEMPLATES_SMALL

from unsloth_clip_lora import FastUnslothLoRALinear, torch_compile_options

def get_dataset_name_for_template(dataset):
    dataset_name = {
        "imagenet_100": "",
        "imagenet": "",
        "std10": "",
        "pets": "pet ",
        "fgvc_aircraft": "aircraft ",
        "cars": "car ",
        "eurosat": "satellite ",
        "dtd": "texture ",
        "flowers102": "flower ",
        "food101": "food ",
        "sun397": "scene ",
        "caltech101": "",
    }[dataset]
    return dataset_name

def lora_replace_attention_layers_unsloth_style(module, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            fast_lora = FastUnslothLoRALinear(child, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            setattr(module, name, fast_lora)
        else:
            lora_replace_attention_layers_unsloth_style(child, lora_r, lora_alpha, lora_dropout)
    return module

def unsloth_patch_clip_attention(model):
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            def make_fast_forward(orig_fwd):
                def fast_forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, **kwargs):
                    return orig_fwd(
                        query, key, value, 
                        key_padding_mask=key_padding_mask, 
                        need_weights=False,
                        attn_mask=attn_mask,
                        **kwargs
                    )
                return fast_forward

            module.forward = types.MethodType(make_fast_forward(module.forward), module)

def custom_checkpoint_forward(self, x):
    for resblock in self.resblocks:
        x = checkpoint(resblock, x, use_reentrant=False)
    return x

class UnslothLoRA_Q_V_Attention(nn.Module):
    def __init__(self, orig_attn: nn.MultiheadAttention, lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.embed_dim = orig_attn.embed_dim
        self.num_heads = orig_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        has_bias = orig_attn.in_proj_bias is not None
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias)

        with torch.no_grad():
            w_q, w_k, w_v = orig_attn.in_proj_weight.chunk(3, dim=0)
            self.q_proj.weight.copy_(w_q)
            self.k_proj.weight.copy_(w_k)
            self.v_proj.weight.copy_(w_v)
            
            if has_bias:
                b_q, b_k, b_v = orig_attn.in_proj_bias.chunk(3, dim=0)
                self.q_proj.bias.copy_(b_q)
                self.k_proj.bias.copy_(b_k)
                self.v_proj.bias.copy_(b_v)
                
        self.out_proj = orig_attn.out_proj

        self.q_proj = FastUnslothLoRALinear(self.q_proj, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.v_proj = FastUnslothLoRALinear(self.v_proj, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, **kwargs):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        tgt_len, bsz, _ = query.shape

        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=0.0
        )

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, self.embed_dim)
        
        output = self.out_proj(attn_output)
        return output, None

def apply_lora_qv_to_clip_transformer(transformer_module, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    for block in transformer_module.resblocks:
        orig_attn = block.attn
        block.attn = UnslothLoRA_Q_V_Attention(
            orig_attn, 
            lora_r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout
        )
    return transformer_module

class CLIP(nn.Module):
    def __init__(
        self, 
        dataset,
        is_lora_image,
        is_lora_text,
        clip_download_dir="model_clip",
        clip_version="ViT-B/16",
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = get_dataset_name_for_template(dataset)
        self.is_lora_image = is_lora_image
        self.is_lora_text = is_lora_text
        self.clip_version = clip_version

        # TODO: change the number of templates
        self.templates = TEMPLATES_SMALL

        self.clip, _ = clip.load(clip_version, device="cpu", download_root=clip_download_dir)

        unsloth_patch_clip_attention(self.clip)

        # visual model
        if is_lora_image and self.clip_version != "RN50":
            self.clip.visual.transformer = apply_lora_qv_to_clip_transformer(
                self.clip.visual.transformer, lora_r=16, lora_alpha=32, lora_dropout=0.1
            )

        # text model
        if is_lora_text:
            self.clip.transformer = apply_lora_qv_to_clip_transformer(
                self.clip.transformer, lora_r=16, lora_alpha=32, lora_dropout=0.1
            )

        self.register_buffer("tokenized_text", self.tokenize_text())

        self.apply_unsloth_compilation()
        self.set_learnable_params()

    def apply_unsloth_compilation(self):
        print("Compiling model parts using Unsloth's torch.compile options...")

        if hasattr(self.clip, "transformer"):
            for block in self.clip.transformer.resblocks:
                block.mlp = torch.compile(block.mlp, dynamic=True, options=torch_compile_options)
                
        if hasattr(self.clip.visual, "transformer"):
            for block in self.clip.visual.transformer.resblocks:
                block.mlp = torch.compile(block.mlp, dynamic=True, options=torch_compile_options)

    def _compile_resblocks(self):
        print("Compiling ResBlocks for faster LayerNorm/MLP (Unsloth style)...")
        for block in self.clip.transformer.resblocks:
            block.mlp = torch.compile(block.mlp)
        if self.clip_version != "RN50":
            for block in self.clip.visual.transformer.resblocks:
                block.mlp = torch.compile(block.mlp)

#     @staticmethod
    def tokenize_text(self):
        print("Tokenizing text...")
        texts = []
        for classname in SUBSET_NAMES[self.dataset]:
            class_texts = [template.format(self.dataset_name, classname) for template in self.templates]
            texts.append(clip.tokenize(class_texts))
        
        # Shape: [n_classes, n_templates, ctx_length]
        return torch.stack(texts)

    def set_learnable_params(self):
        # turn off all parameters
        self.clip.requires_grad_(False)

        # learnable parameters for the visual model
        if self.is_lora_image:
            if self.clip_version != "RN50":
                for name, p in self.clip.visual.named_parameters():
                    if "lora_" in name:
                        p.requires_grad = True
            else:
                self.clip.visual.requires_grad_(True)
                
        # learnable parameters for the text model
        if self.is_lora_text:
            for name, p in self.clip.transformer.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

    def learnable_params(self):
#         return [{"name": "all", "params": [p for p in self.clip.parameters() if p.requires_grad]}]
        return [p for p in self.clip.parameters() if p.requires_grad]

    def forward_image(
        self,
        x: torch.Tensor,
    ):
        image_feats = self.clip.visual(x)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        return image_feats

    def forward_text(self, tokenized_text):
        n_classes, n_prompts, n_token = tokenized_text.size()
        tokenized_text = tokenized_text.view(-1, n_token)

        context = torch.enable_grad() if self.is_lora_text else torch.inference_mode()
        
        with context:
            text_feats = self.clip.encode_text(tokenized_text)

        text_feats = F.normalize(text_feats, dim=-1)

        text_feats = text_feats.view(n_classes, n_prompts, -1).mean(dim=1)
        text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(
        self,
        x: torch.Tensor,
        tokenized_text: torch.Tensor = None,
        output_features: bool = False,
        n_spc_samples: int = 100,
        **kwargs,
    ):
        if tokenized_text is None:
            tokenized_text = self.tokenized_text

        with torch.autocast(device_type=x.device.type, dtype=torch.float16):
            image_feats = self.forward_image(x)
            text_feats = self.forward_text(tokenized_text)

            logit_scale = self.clip.logit_scale.exp()

            if text_feats.ndim == 2:
                logits_per_image = logit_scale * (image_feats @ text_feats.t())
            else:
                # image_feats: [B, Dim] -> [B, 1, Dim]
                # text_feats: [B, Classes, Dim] -> [B, Dim, Classes]
                logits_per_image = logit_scale * torch.bmm(
                    image_feats.unsqueeze(1), 
                    text_feats.transpose(1, 2)
                ).squeeze(1)

        if output_features:
            return {
                "logits": logits_per_image, 
                "image_feats": image_feats.float(),
                "text_feats": text_feats.float(),
            }

        return logits_per_image