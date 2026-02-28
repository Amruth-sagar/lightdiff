import torch
import pickle
import argparse
from tqdm import tqdm

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + (eps * std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_data_file', type=str, required=True)
    parser.add_argument('--only_mu', action="store_true")
    args = parser.parse_args()

    with open(args.latent_data_file, 'rb') as infile:
        latent_data = pickle.load(infile)

    all_latents = []
    for data in tqdm(latent_data):
        mu = data["mu"]
        logvar = data["logvar"]
        if args.only_mu:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        all_latents.append(z.flatten())

    all_latents_tensor = torch.cat(all_latents)
    global_std = all_latents_tensor.std()

    print(f"Global mean: {all_latents_tensor.mean()}\nGlobal std: {all_latents_tensor.std()}\nScaling Factor (1/std):{1/global_std}")

