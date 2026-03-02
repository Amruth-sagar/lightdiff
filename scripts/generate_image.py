from lightdiff.vae.model import VAE
from lightdiff.unet.model import UNet
from lightdiff.diffusion.scheduler import NoiseScheduler
from lightdiff.diffusion.samplers import ddim_sample
from torchvision import transforms
from PIL import Image
import torch
import argparse


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--vae_ckpt_path', type=str, required=True)
    argument_parser.add_argument('--unet_ckpt_path', type=str, required=True)
    argument_parser.add_argument('--noise_type', type=str, required=True)
    argument_parser.add_argument('--save_dir', type=str, required=True)
    argument_parser.add_argument('--image_name_prefix', type=str, required=True)
    argument_parser.add_argument('--device', type=str, default='cuda:0')
    argument_parser.add_argument('--resolution', type=int, default=256)
    argument_parser.add_argument('--num_steps', type=int, default=50)
    argument_parser.add_argument('--num_images', type=int, required=True)
    argument_parser.add_argument('--max_timesteps', type=int, required=True)
    argument_parser.add_argument('--latent_shape', nargs="+", type=int, required=True) # 3 numbers in the format (latent_channels, latent_h, latent_w)
    argument_parser.add_argument('--scaling_factor', type=float, required=True)
    argument_parser.add_argument('--random_seed', type=int, required=True)

    args = argument_parser.parse_args()

    device = torch.device(args.device)


    ckpt_vae = torch.load(args.vae_ckpt_path, map_location=device)
    ckpt_unet = torch.load(args.unet_ckpt_path, map_location=device)

    model_vae = VAE(ckpt_vae['VAE_CFG'])
    model_unet = UNet(ckpt_unet['UNET_CFG'])

    model_vae.load_state_dict(ckpt_vae['model_state_dict'])
    # for unet, we load the ema model
    model_unet.load_state_dict(ckpt_unet['ema_state_dict'])

    model_vae.to(device)
    model_unet.to(device)

    scheduler = NoiseScheduler(args.max_timesteps, "cosine", device=device)

    resolution = args.resolution

    # For transforming images to tensor for vae input 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])
    
    output_latent = ddim_sample(
        model=model_unet, scheduler=scheduler, 
        shape=tuple(args.latent_shape), 
        num_steps=args.num_steps, 
        num_images=args.num_images, 
        random_seed=args.random_seed)
    reconstructed = model_vae.decode(output_latent / args.scaling_factor)
    # brining back from [-1,-1] to [0,1] range
    reconstructed_rescaled = (reconstructed +1 )/2
    reconstructed_rescaled = reconstructed_rescaled.clamp(0,1)

    for i in range(reconstructed_rescaled.shape[0]):
        reconstructed_pil_image = transforms.functional.to_pil_image(reconstructed_rescaled[i])
        reconstructed_pil_image.save(f'{args.save_dir}/{args.image_name_prefix}_seed_{args.random_seed}_steps_{args.num_steps}_{i}.png')

    
