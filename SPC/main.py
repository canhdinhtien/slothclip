import os
import gc
import torch
import random
import numpy as np

from itertools import cycle
from torch.utils.tensorboard import SummaryWriter

from unsloth_loss import fast_clip_cross_entropy_loss
from models.clip import CLIP
from utils import cosine_scheduler, get_acc
from data.data import get_data_loader, get_synth_train_data_loader

def fix_random_seed(seed=22):
    """
    Fix random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(
    model, 
    opt_h, 
    scaler, 
    step, 
    fewshot_train_loader, 
    loader_iter_G, 
    lamda1,
    lr_schedule_values, 
    writer, 
    device, 
    dtype_clip
):
    model.train()
    
    for real_images, real_labels in fewshot_train_loader:
        if step < len(lr_schedule_values):
            current_lr = lr_schedule_values[step]
            for param_group in opt_h.param_groups:
                param_group["lr"] = current_lr
                
            writer.add_scalar("Train/Learning_Rate", current_lr, step)

        synth_images, synth_labels = next(loader_iter_G)
        
        bs_real = real_images.size(0)
        bs_synth = synth_images.size(0)
        
        all_images = torch.cat([real_images, synth_images], dim=0).to(device, dtype=dtype_clip)
        all_labels = torch.cat([real_labels, synth_labels], dim=0).to(device)

        opt_h.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            logits_all = model(all_images)
            
            logits_real = logits_all[:bs_real]
            logits_synth = logits_all[bs_real:]
            
            loss_real = fast_clip_cross_entropy_loss(logits_real, all_labels[:bs_real])
            loss_synth = fast_clip_cross_entropy_loss(logits_synth, all_labels[bs_real:])

            total_loss = loss_real + lamda1 * loss_synth
            
            log_metrics = {
                "br": loss_real.item(), 
                "bs": loss_synth.item()
            }

        scaler.scale(total_loss).backward()
        scaler.step(opt_h)
        scaler.update()

        writer.add_scalar("Loss_Batch/Real_Base", log_metrics["br"], step)
        writer.add_scalar("Loss_Batch/Synth_Base", log_metrics["bs"], step)
        writer.add_scalar("Loss_Batch/Total", total_loss.item(), step)

        del all_images, all_labels, logits_all, logits_real, logits_synth, loss_real, loss_synth, total_loss

        step += 1

    return step

def main():
    fix_random_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "dtd"
    model_type = "clip"
    n_samples_per_class = 16
    n_synth_per_class = 64
    n_epochs = 50
    batch_size = 64
    eval_batch_size = 256
    synth_train_data_dir = "./synthetic_data"

    lamda1_s = [0.1, 0.2]

    fewshot_train_loader, test_loader = get_data_loader(
        real_train_data_dir="",
        real_test_data_dir="",
        dataset=dataset,
        bs=batch_size,
        eval_bs=eval_batch_size,
        n_img_per_cls=n_samples_per_class,
        model_type=model_type
    )

    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=synth_train_data_dir,
        bs=batch_size,
        n_img_per_cls=n_synth_per_class,
        dataset=dataset,
        model_type=model_type
    )
    print(f"Number of few-shot training samples: {len(fewshot_train_loader.dataset)}")
    print(f"Number of synthetic training samples: {len(synth_train_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    
    for lamda1 in lamda1_s:
        exp_name = f"lda1_{lamda1}"
        print('-------------------------------------------------')

        model = CLIP(
            dataset="dtd",
            is_lora_image=True,
            is_lora_text=True,
            clip_download_dir="model_clip",
            clip_version="ViT-B/16",
        ).to(device)

        dtype_clip = model.clip.visual.conv1.weight.dtype

        log_dir_path = f"runs/{exp_name}"
        ckpt_path = f"checkpoints/{exp_name}"
        os.makedirs(log_dir_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)

        writer = SummaryWriter(log_dir=log_dir_path)
        opt_h = torch.optim.AdamW(model.learnable_params(), lr=1e-4)

        scaler = torch.amp.GradScaler('cuda')

        step = 0
        loader_iter_G = cycle(synth_train_loader)

        iters_per_epoch = len(fewshot_train_loader) 
        
        lr_schedule_values = cosine_scheduler(
            base_value=1e-4,             
            final_value=1e-6,               
            epochs=n_epochs,                
            niter_per_ep=iters_per_epoch,   
            warmup_epochs=5,                 
            start_warmup_value=1e-6          
        )

        for epoch in range(n_epochs):
            step = train_one_epoch(
                model=model, 
                opt_h=opt_h, 
                scaler=scaler, 
                step=step, 
                fewshot_train_loader=fewshot_train_loader, 
                loader_iter_G=loader_iter_G, 
                lamda1=lamda1,
                lr_schedule_values=lr_schedule_values,
                writer=writer, 
                device=device, 
                dtype_clip=dtype_clip
            )

            model.eval()
            with torch.no_grad():
                train_acc = get_acc(model, fewshot_train_loader)
                test_acc = get_acc(model, test_loader)

            writer.add_scalar("Epoch/Train_Accuracy", train_acc, global_step=epoch)
            writer.add_scalar("Epoch/Eval_Accuracy", test_acc, global_step=epoch)
            
            print(f"Epoch {epoch} | Train ACC: {train_acc*100:.2f}% | Test ACC: {test_acc*100:.2f}%")

        writer.close()
        gc.collect()
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()