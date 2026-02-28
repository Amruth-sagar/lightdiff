from glob import glob
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from lightdiff.vae.model import VAE
import pickle
import argparse
import torch
import os

@torch.no_grad()
def convert_imgs_to_latents(
    fair_face_dir, 
    ffhq_dir,
    vae_model,
    save_dir,
    device,
    resolution=256,
    batch_size=64
):
    all_fair_face_paths = glob(f'{fair_face_dir}/**/*.jpg', recursive=True)
    all_ffhq_paths = glob(f'{ffhq_dir}/**/*.png', recursive=True)

    all_paths = all_fair_face_paths + all_ffhq_paths 

    print(f"Total n.of images : {len(all_paths)}")

    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        # [0, 1] --> [-1, -1] ( (tensor - 0.5) / 0.5 happens each channel)
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    latents = []
    vae_model.eval()

    for i in tqdm(range(0, len(all_paths), batch_size)):
        batch_of_paths = all_paths[i:i+batch_size]
        batch_of_images = []
        valid_paths_of_batch = []
        for path in batch_of_paths:
            try:
                img = Image.open(path).convert("RGB")
                # (3, H, W) -> (1, 3, H, W)
                img = transform(img).unsqueeze(0)

                batch_of_images.append(img)
                valid_paths_of_batch.append(path)

            except:
                continue

        # If it failed to read any
        if not batch_of_images:
            continue

        input_tensor = torch.concatenate(batch_of_images, dim=0).to(device)

        with torch.no_grad():
            output = vae_model(input_tensor)
            mu_s = output["mu"].cpu()
            logvar_s = output["logvar"].cpu()


        # CLONE! if not we end up storing the whole batch for every image!
        # if mu_s is (batch, dim, h, w), then mu_s[j] is just a view of the whole tensor, 
        # and the mu_s[j].untyped_storage().nbytes() is same as mu_s.untyped_storage().nbytes()
        for j in range(len(valid_paths_of_batch)):
            temp = {
                "path": valid_paths_of_batch[j], 
                "mu": mu_s[j].clone(),
                "logvar": logvar_s[j].clone()
            }
            latents.append(temp)


    
    with open(f'{save_dir}/latent_data.pkl', 'wb') as outfile:
        pickle.dump(latents, outfile)



if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--ffhq_dir', type=str, required=True)
    argument_parser.add_argument('--fair_face_dir', type=str, required=True)
    argument_parser.add_argument('--save_dir', type=str, required=True)
    argument_parser.add_argument('--ckpt_path', type=str, required=True)

    args = argument_parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda:0")
    
    ckpt = torch.load(args.ckpt_path)
    model = VAE(ckpt['VAE_CFG'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    convert_imgs_to_latents(
        fair_face_dir=args.fair_face_dir,
        ffhq_dir = args.ffhq_dir,
        vae_model=model,
        save_dir=args.save_dir,
        device=device
    )
    
    
    
    
