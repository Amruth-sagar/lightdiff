from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from torchvision import transforms
from PIL import Image
import random
import torch
import pickle

class DatasetForVAE(Dataset):
    def __init__(self, ffhq_dir, fair_face_dir, split="train", train_val=[0.95,0.05], resolution=256, random_seed=42):
        super().__init__()

        assert sum(train_val) == 1, "Train and val split percentages should add up to one"

        all_fair_face_paths = glob(f'{fair_face_dir}/**/*.jpg', recursive=True)
        all_ffhq_256_paths = glob(f'{ffhq_dir}/**/*.png', recursive=True)

        print(f"Total number of images, before split: {len(all_fair_face_paths)+len(all_ffhq_256_paths)}")

        rng = random.Random(random_seed)

        rng.shuffle(all_fair_face_paths)
        rng.shuffle(all_ffhq_256_paths)

        if split == "train":
            art_paths = all_fair_face_paths[:int(len(all_fair_face_paths)*train_val[0])] 
            face_paths = all_ffhq_256_paths[:int(len(all_ffhq_256_paths)*train_val[0])]
        elif split == "val":
            art_paths = all_fair_face_paths[int(len(all_fair_face_paths)*train_val[0]):]
            face_paths = all_ffhq_256_paths[int(len(all_ffhq_256_paths)*train_val[0]):]
        else:
            raise ValueError("split should either be \'train\' or \'val\'")

        self.face_paths = face_paths
        self.art_paths = art_paths
        self.all_paths = self.face_paths + self.art_paths

        self.transform = transforms.Compose([
            transforms.Resize(resolution + 16),
            transforms.RandomResizedCrop(resolution, scale=(0.9,1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # [0, 1] --> [-1, -1] ( (tensor - 0.5) / 0.5 happens each channel)
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, index):
        path = self.all_paths[index]

        try: 
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except:
            return self.__getitem__((index + 1) % len(self))
        
        return img
    

def get_dataloader_vae(dataset, batch_size=32, num_workers=4, is_train=True, is_ddp = True, dist_sampler_seed=0):

    loader_args = {
        "batch_size": batch_size,
        "pin_memory": True,
        "num_workers": num_workers
    }

    if is_ddp:
        sampler = DistributedSampler(
            dataset, 
            shuffle=is_train, 
            seed=dist_sampler_seed
        )
        shuffle = False
    else:
        sampler = None
        shuffle = is_train
    
    return DataLoader(
        dataset, 
        sampler=sampler, 
        shuffle=shuffle,
        **loader_args
    )



class DatasetForDiff(Dataset):
    def __init__(self, 
                 data_file, scaling_factor, noise_scheduler, split="train", 
                 train_val=[0.95,0.05], num_timesteps=1000, random_seed=42):
        super().__init__()

        with open(data_file, 'rb') as infile:
            full_data = pickle.load(infile)

        rng = random.Random(random_seed)
        rng.shuffle(full_data)

        num_samples = len(full_data)
        split_idx = int(num_samples * train_val[0])

        if split == "train":
            self.latent_data = full_data[:split_idx]
        else:
            self.latent_data = full_data[split_idx:]

        self.split = split
        self.noise_scheduler = noise_scheduler
        self.scaling_factor = scaling_factor
        self.num_timesteps = num_timesteps
        self.seed = random_seed

    def __len__(self):
        return len(self.latent_data)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)
  
    def __getitem__(self, idx):
        data = self.latent_data[idx]
        
        gen = torch.Generator()

        if self.split == "val":
            gen.manual_seed(self.seed + idx)
        else:
            gen.manual_seed(torch.seed())

        mu, logvar = data["mu"], data["logvar"]

        # TODO: remove if mu * scaling_factor works
        # sampled_latent = self.reparameterize(mu, logvar) * self.scaling_factor
        sampled_latent = mu * self.scaling_factor

        t = torch.randint(0, self.num_timesteps, (1,), generator=gen).long()
        noise = torch.randn(sampled_latent.shape, generator=gen)
        
        noised_latent = self.noise_scheduler.add_noise(sampled_latent, noise, t)

        return {
            "noised_latent": noised_latent.squeeze(0) if noised_latent.dim() == 4 else noised_latent,
            "orig_noise": noise,
            "timesteps": t.squeeze()
        }

def diff_collate_fn(batch):

    noised_latents = torch.stack([item['noised_latent'] for item in batch])             # (batch, lc, lh, lw)
    orig_noises = torch.stack([item['orig_noise'] for item in batch])                   # (batch, lc, lh, lw)
    timesteps = torch.tensor([item['timesteps'] for item in batch], dtype=torch.long)   # (batch,)

    return {
        "noised_latents": noised_latents,
        "orig_noises": orig_noises,
        "timesteps": timesteps
    }



def get_dataloader_diff(dataset, batch_size=32, is_train=True, is_ddp=True, num_workers=4, dist_sampler_seed=0):

    loader_args = {
        "batch_size": batch_size,
        "collate_fn": diff_collate_fn,
        "pin_memory": True,
        "num_workers": num_workers
    }


    if is_ddp:
        sampler = DistributedSampler(
            dataset, 
            shuffle=is_train, 
            seed=dist_sampler_seed
        )
        shuffle = False
    else:
        sampler = None
        shuffle = is_train
    
    return DataLoader(
        dataset, 
        sampler=sampler, 
        shuffle=shuffle,
        **loader_args
    )