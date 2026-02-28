from lightdiff.data.dataset import DatasetForVAE, get_dataloader_vae
from lightdiff.vae.model import VAE
from lightdiff.vae.loss import VAEloss
import torch
import argparse
import random
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim import AdamW

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

def save_checkpoint(checkpoint_name, ckpt_dir, model, optimizer, optimizer_steps, global_step, prev_val_loss, args, VAE_CFG):
    save_path = f'{ckpt_dir}/{checkpoint_name}.pth'
    torch.save({
        "model_state_dict": model.module.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "optimizer_steps": optimizer_steps,
        "prev_val_loss": prev_val_loss,
        "args": vars(args),
        "VAE_CFG": VAE_CFG,
    }, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--metric_log_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--kl_beta", type=float, required=True)
    parser.add_argument("--lpips_gamma", type=float, required=True)
    parser.add_argument("--ffhq_dir", type=str, required=True)
    parser.add_argument("--fair_face_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
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

    VAE_CFG = {
        "num_enc_blocks": 3, 
        "num_dec_blocks": 3, 
        "enc_channel_args": [64, 128, 256, 512],
        "dec_channel_args": [512, 256, 128, 64],
        "num_conv_per_enc_block": [3, 4, 4], 
        "num_conv_per_dec_block": [3, 4, 4], 
        "latent_channels": 4, 
        "final_activation": "tanh"
    }

    model = VAE(VAE_CFG).to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = VAEloss(device=device)

    train_dataset = DatasetForVAE(ffhq_dir=args.ffhq_dir, fair_face_dir=args.fair_face_dir, split="train", resolution=256, random_seed=args.random_seed)
    val_dataset = DatasetForVAE(ffhq_dir=args.ffhq_dir, fair_face_dir=args.fair_face_dir, split="val", resolution=256, random_seed=args.random_seed)

    train_dataloader = get_dataloader_vae(train_dataset, batch_size=args.batch_size, is_train=True, is_ddp=True)
    val_dataloader = get_dataloader_vae(val_dataset, batch_size=args.batch_size, is_train=False, is_ddp=True)

    global_step = 0
    optimizer_steps = 0
    prev_val_loss = torch.inf
    optimizer.zero_grad(set_to_none=True)

    if args.resume_from_ckpt is not None:
        map_location = {'cuda:0': f'cuda:{local_rank}'}
        ckpt = torch.load(args.resume_from_ckpt, map_location)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        global_step = ckpt['global_step']
        optimizer_steps = ckpt['optimizer_steps']
        prev_val_loss = ckpt['prev_val_loss']

    model.train()

    beta, gamma = args.kl_beta, args.lpips_gamma

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"EPOCH: {epoch + 1} ...\n")
            
        train_dataloader.sampler.set_epoch(epoch)
        
        for images in train_dataloader: 
            images = images.to(device)
            is_accumulating = (global_step + 1) % args.accum_grad_after_k_steps != 0

            # gradient accumulation
            if is_accumulating:
                with model.no_sync():
                    model_output = model(images)
                    loss_dict = criterion(
                         x=images, x_reconstructed=model_output['reconstructed'], 
                         mu=model_output['mu'], logvar=model_output['logvar'],
                         beta=beta, gamma=gamma
                    )
                    loss = loss_dict['total'] / args.accum_grad_after_k_steps
                    loss.backward()
            else:
                model_output = model(images)
                loss_dict = criterion(
                     x=images, x_reconstructed=model_output['reconstructed'], 
                     mu=model_output['mu'], logvar=model_output['logvar'],
                     beta=beta, gamma=gamma
                )
                loss = loss_dict['total'] / args.accum_grad_after_k_steps
                loss.backward()

                # logging
                if rank == 0:
                    train_loss = loss.detach() * args.accum_grad_after_k_steps
                    writer.add_scalar("loss/train", train_loss, global_step+1)
                    writer.add_scalar("loss/train_kl", loss_dict['kl'].detach(), global_step+1)
                    writer.add_scalar("loss/train_lpips", loss_dict['lpips'].detach(), global_step+1)
                    writer.add_scalar("loss/train_mse", loss_dict['mse'].detach(), global_step+1)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            # we do validation after few optimizer steps ( an optimizer step is k gradient accumulations and a synced backward)
            if optimizer_steps > 0 and optimizer_steps % args.val_after_k_optim_steps == 0:
                dist.barrier()
                model.eval()

                local_val_loss = 0.0

                with torch.no_grad():
                    for val_images in val_dataloader:
                        val_images = val_images.to(device)
                        model_output = model(val_images)
                        loss_dict = criterion(
                                x=val_images, x_reconstructed=model_output['reconstructed'], 
                                mu=model_output['mu'], logvar=model_output['logvar'],
                                beta=beta, gamma=gamma
                            )
                        local_val_loss += loss_dict['total'].item()

                local_val_loss /= len(val_dataloader)

                val_loss_tensor = torch.tensor(local_val_loss).to(device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

                global_val_loss = val_loss_tensor.item() / world_size

                if rank == 0:
                    writer.add_scalar("loss/val", global_val_loss, global_step+1)
                    if prev_val_loss > global_val_loss:
                        prev_val_loss = global_val_loss
                        save_checkpoint("VAE_best_val", args.ckpt_dir, model, optimizer, optimizer_steps, global_step, prev_val_loss, args, VAE_CFG)
                    
                
                model.train()
                dist.barrier()
                optimizer_steps = 0
                

            if (global_step + 1) % args.save_ckpt_after_k_steps == 0 and (global_step + 1) >= args.start_saving_ckpt_after_k_steps:
                dist.barrier()
                if rank == 0:
                    checkpoint_name = f"VAE_step_{global_step+1}_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    save_checkpoint(checkpoint_name, args.ckpt_dir, model, optimizer, optimizer_steps, global_step, prev_val_loss, args, VAE_CFG)
                dist.barrier()
            
            global_step += 1

    cleanup_distributed()
    if rank == 0 and writer is not None:
        writer.close()

if __name__ == "__main__":
    main()