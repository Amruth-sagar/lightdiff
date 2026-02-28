from lightdiff.data.dataset import DatasetForDiff, get_dataloader_diff
from lightdiff.diffusion.scheduler import NoiseScheduler
from lightdiff.unet.model import UNet
import torch
import argparse
import random
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim import AdamW
from copy import deepcopy


# EMA copy of the UNet

class EMAModel:
    def __init__(self, model, decay=0.999):
        # if DDP, should access .module of model
        self.ema_model = deepcopy(model.module if hasattr(model, 'module') else model)
        self.ema_model.eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_params = model.module.parameters()
        for ema_v, model_v in zip(self.ema_model.parameters(), model_params):
            ema_v.copy_((self.decay * ema_v) + ((1.0 - self.decay) * model_v)) 

        model_buffers = dict(model.module.named_buffers())
        for name, ema_b in self.ema_model.named_buffers():
            model_b = model_buffers[name].detach()
            ema_b.copy_(model_b)

def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def make_deterministic_minimal(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(checkpoint_name, ckpt_dir, model, ema_model, optimizer, optimizer_steps, global_step, prev_val_loss, args, UNET_CFG):
    save_path = f'{ckpt_dir}/{checkpoint_name}.pth'
    torch.save({
        "model_state_dict": model.module.state_dict(),
        "ema_state_dict": ema_model.ema_model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "optimizer_steps": optimizer_steps,
        "prev_val_loss": prev_val_loss,
        "args": vars(args),
        "UNET_CFG": UNET_CFG,
    }, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--metric_log_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--decay", type=float, default=0.999)
    parser.add_argument("--scaling_factor", type=float, required=True)
    parser.add_argument("--latent_data_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--total_timesteps", type=int, required=True)
    parser.add_argument("--noise_scheduler_type", type=str, required=True)
    parser.add_argument("--resume_from_ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--accum_grad_after_k_steps", type=int, required=True)
    parser.add_argument("--val_after_k_optim_steps", type=int, required=True)
    parser.add_argument("--start_saving_ckpt_after_k_steps", type=int, default=0)
    parser.add_argument("--save_ckpt_after_k_steps", type=int, required=True)

    args = parser.parse_args()
    make_deterministic_minimal(args.random_seed)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.metric_log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.metric_log_dir) if rank == 0 else None

    UNET_CFG = {
        "num_down_layers": 3, 
        "num_up_layers": 3, 
        "down_channel_args": [128, 256, 512, 1024], 
        "up_channel_args": [1024, 512, 256, 128], 
        "latent_channels": 16, 
        "num_res_blocks_per_level": 3, 
        "time_dim": 512,
    }

    model = UNet(UNET_CFG).to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    ema_model = EMAModel(model, decay=args.decay)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_dataset = DatasetForDiff(
        data_file=args.latent_data_file,
        scaling_factor=args.scaling_factor, 
        noise_scheduler=NoiseScheduler(
            num_timesteps=args.total_timesteps,
            schedule_type=args.noise_scheduler_type,
            device="cpu"        
        ),
        split="train",
        num_timesteps=args.total_timesteps,
        random_seed=args.random_seed
    )
    val_dataset = DatasetForDiff(
        data_file=args.latent_data_file,
        scaling_factor=args.scaling_factor, 
        noise_scheduler=NoiseScheduler(
            num_timesteps=args.total_timesteps,
            schedule_type=args.noise_scheduler_type,
            device="cpu"
        ),
        split="val",
        num_timesteps=args.total_timesteps,
        random_seed=args.random_seed
    )
    train_dataloader = get_dataloader_diff(train_dataset, batch_size=args.batch_size, is_train=True, is_ddp=True)
    val_dataloader = get_dataloader_diff(val_dataset, batch_size=args.batch_size, is_train=False, is_ddp=True)

    global_step = 0
    optimizer_steps = 0
    prev_val_loss = torch.inf
    optimizer.zero_grad(set_to_none=True)

    if args.resume_from_ckpt is not None:
        print(f"Resuming from ckpt {args.resume_from_ckpt}")
        map_location = {'cuda:0': f'cuda:{local_rank}'}
        ckpt = torch.load(args.resume_from_ckpt, map_location)
        model.module.load_state_dict(ckpt['model_state_dict'])
        ema_model.ema_model.load_state_dict(ckpt['ema_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        global_step = ckpt['global_step']
        optimizer_steps = ckpt['optimizer_steps']
        prev_val_loss = ckpt['prev_val_loss']

    model.train()

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"EPOCH: {epoch + 1} ...\n")
            
        train_dataloader.sampler.set_epoch(epoch)

        for batch in train_dataloader: 
            x = batch['noised_latents'].to(device)
            t = batch['timesteps'].to(device)
            orig_noises = batch['orig_noises'].to(device)
            
            is_accumulating = (global_step + 1) % args.accum_grad_after_k_steps != 0

            if is_accumulating:
                with model.no_sync():
                    pred_noise = model(x, t)
                    loss = F.mse_loss(pred_noise, orig_noises)
                    loss = loss / args.accum_grad_after_k_steps
                    loss.backward()
            else:
                pred_noise = model(x, t)
                loss = F.mse_loss(pred_noise, orig_noises)
                loss = loss / args.accum_grad_after_k_steps
                loss.backward()

                if rank == 0:
                    train_loss = loss.detach() * args.accum_grad_after_k_steps
                    writer.add_scalar("loss/train", train_loss, global_step+1)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

                # Updating our ema model after an optimizer update.
                ema_model.update(model)

            if optimizer_steps > 0 and optimizer_steps % args.val_after_k_optim_steps == 0:

                dist.barrier()
                model.eval()

                local_val_loss = 0.0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_x = val_batch['noised_latents'].to(device)
                        val_t = val_batch['timesteps'].to(device)
                        orig_noises_val = val_batch['orig_noises'].to(device)

                        pred_noise = model(val_x, val_t)
                        loss = F.mse_loss(pred_noise, orig_noises_val)
                        local_val_loss += loss.item()

                    
                local_val_loss /= len(val_dataloader)
                val_loss_tensor = torch.tensor(local_val_loss).to(device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

                global_val_loss = val_loss_tensor.item() / world_size

                if rank == 0:
                    writer.add_scalar("loss/val", global_val_loss, global_step+1)
                
                    if prev_val_loss > global_val_loss:
                        prev_val_loss = global_val_loss
                        save_checkpoint("UNET_best_val", args.ckpt_dir, model, ema_model, optimizer, optimizer_steps, global_step, prev_val_loss, args, UNET_CFG)

                 
                model.train()
                dist.barrier()

                optimizer_steps = 0

            if (global_step + 1) % args.save_ckpt_after_k_steps == 0 and (global_step + 1) >= args.start_saving_ckpt_after_k_steps:
                dist.barrier()
                if rank == 0:
                    checkpoint_name = f"UNET_step_{global_step+1}_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    save_checkpoint(checkpoint_name, args.ckpt_dir, model, ema_model, optimizer, optimizer_steps, global_step, prev_val_loss, args, UNET_CFG)
                dist.barrier()
            
            global_step += 1

    cleanup_distributed()
    if rank == 0 and writer is not None:
        writer.close()

if __name__ == "__main__":
    main()