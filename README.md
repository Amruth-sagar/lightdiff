# :robot: `lightdiff` Latent Diffusion from-scratch 
This is a lean, from-scratch implementation that trades the abstraction in massive frameworks for architectural clarity, while still handling the heavy lifting of distributed multi-GPU training and face generation. This repo is designed as a guided tour through the entire pipeline: from compressing pixels into a latent manifold with a VAE, to prioritizing the type of latents we need, to orchestrating denoising with a UNet. You can get your hands dirty with the math, vary beta and LPIPS weights as per your goal, and see exactly how each hyperparameter ripples through to the final pixels.

# :wrench: Installation

## :clipboard: Prerequisites
- Python 3.12 or higher
- Pytorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

1. **Create a virtual environment**:  Create a virtual environment 
```
conda create -n ldiff python=3.13
conda activate ldiff
```

2. **Instally PyTorch with CUDA** in the virtual environment:  For windows and linux
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
For MacOS, 
```
pip3 install torch torchvision
```

3. **Clone this repo and install the dependencies**
```
git clone https://github.com/Amruth-sagar/lightdiff.git
cd llm_from_scratch
pip install -e .
```

# :package: Repository structure and overview

This repository is organized into a modular library called lightdiff and a set of executable scripts. While the scripts handle the "how" of training and generation, the lightdiff directory contains the "what"—the core architecture, schedulers, and data-handling logic.

```text
.
├── lightdiff/                # Core Library
│   ├── data/
│   │   └── dataset.py        # Dataset classes for VAE (Images) and UNet (Latents)
│   ├── diffusion/
│   │   ├── samplers.py       # DDIM sampling logic
│   │   └── scheduler.py      # NoiseScheduler (Linear & Cosine support)
│   ├── unet/
│   │   ├── blocks.py         # ResNet blocks, Attention, and Down/Up layers
│   │   └── model.py          # Full UNet architecture
│   └── vae/
│       ├── blocks.py         # Encoder/Decoder blocks
│       ├── loss.py           # Combined VAE loss (MSE + KL + LPIPS)
│       └── model.py          # VAE architecture
├── train_vae.py              # Script: Distributed VAE training
├── preprocess_latents.py     # Script: Batch encoding images to latents
├── calculate_scaling.py      # Script: Latent statistics & scaling calculation
├── train_unet.py             # Script: Distributed UNet training (with EMA)
└── generate.py               # Script: DDIM Inference/Generation
```


# :muscle: Training VAE
## :inbox_tray: Downloading datasets
For training the VAE, the following datasets were used.

- **FFHQ-256**
- **Fair Face**

## :gear: Training the model
The script `scripts/trianvae.py` handles the distributed training and validation VAE. It is optimized for high-resolution image reconstruction using a combination of MSE, KL-divergence, and LPIPS perceptual losses.

The model architecture is defined by the `VAE_CFG` dictionary within the main function. While the script comes with defaults, you can modify this dictionary to change the model's capacity and latent space. If you want to change the latent space size or the depth of the network, modify the following parameters in the code:

```python
VAE_CFG = {
    "num_enc_blocks": 3,                     # each block reduces the resolution by half, determining latent resolution in the end.
    "num_dec_blocks": 3, 
    "enc_channel_args": [64, 128, 256, 512], # Channel progression
    "dec_channel_args": [512, 256, 128, 64],
    "num_conv_per_enc_block": [3, 4, 4],
    "num_conv_per_dec_block": [3, 4, 4], 
    "latent_channels": 4,                    # Change this for larger/smaller latent bottlenecks
    "final_activation": "tanh"               # Supports "tanh" or "sigmoid". keep it 'tanh' as lpips loss is used.
}
```

To launch the training script, run

```bash
torchrun --nproc_per_node=NUM_GPUS scripts/trianvae.py \
    --random_seed 42 \
    --lr 1e-4 \
    --kl_beta 0.001 \
    --lpips_gamma 0.5 \
    --batch_size 16 \
    --epochs 50 \
    --ffhq_dir /path/to/ffhq \
    --fair_face_dir /path/to/fairface \
    --metric_log_dir ./logs \
    --ckpt_dir ./checkpoints \
    --accum_grad_after_k_steps 4 \
    --val_after_k_optim_steps 32 \
    --save_ckpt_after_k_steps 2000
```
Balance the $\beta$ and LPIPS weights to match your desired trade-off between latent space regularity and visual reconstruction quality.

# :file_folder: Preprocessing and saving latents to trian UNet
The script `script/preprocess_imgs_to_mu_logvar.py` facilitates Latent Diffusion by pre-calculating and caching the latent representations of the image datasets. By encoding images into mu and logvar tensors ahead of time, you significantly reduce the computational overhead during UNet training, as the VAE encoder no longer needs to run at every step. 

Run the script by pointing it to your image directories and your trained VAE checkpoint:

```bash
python script/preprocess_imgs_to_mu_logvar.py \
    --ffhq_dir /path/to/ffhq \
    --fair_face_dir /path/to/fairface \
    --ckpt_path ./checkpoints/VAE_best_val.pth \
    --save_dir ./preprocessed_data
```

## :triangular_ruler: Calculating Latent Scaling Factor

Before starting UNET training, you have to normalize the latent space. This script processes the previously saved pickle file to calculate the Global Standard Deviation and the corresponding Scaling Factor ($1/\sigma$) that will be multiplied to the latents, so that they are closer to $\mathcal{N}(0,\mathbb{I})$.

Scaling the latents to have unit variance (std = 1) is a critical step in Latent Diffusion Models (LDM), as it ensures the initial noise levels and the UNET's weight initialization are compatible with the distribution of the encoded data.

- Mu-Only Mode: Option to calculate statistics using only the mean ($\mu$), which is often used to get a "clean" estimate of the latent center.
- Global Statistics: Flattens all latent tensors (sampled from the saved mu and logvars) to compute a single, dataset-wide mean and standard deviation.

Run the script `scripts/calculate_scaling_factor.py` by pointing it to the .pkl file generated during the preprocessing step:

```bash
python scripts/calculate_scaling_factor.py \
    --latent_data_file ./preprocessed_data/latent_data.pkl
```
To calculate using only the mean vectors (ignoring variance):

```bash
python scripts/calculate_scaling_factor.py \
    --latent_data_file ./preprocessed_data/latent_data.pkl \
    --only_mu
```


# :muscle: Training your UNet
The script `script/trainunet.py` performs the core diffusion training by training a Resnet style UNet to predict noise added to the precomputed VAE latents. It implements modern diffusion training techniques, such as maintaining an Exponential Moving Average (EMA) "shadow copy" of the model, with decay of 0.999 for example.

Similar to VAE, you can customize the architecture by modifying the `UNET_CFG` in the main method to adjust the capacity of the model.

```python
UNET_CFG = {
    "num_down_layers": 3, 
    "num_up_layers": 3, 
    "down_channel_args": [128, 256, 512, 1024], 
    "up_channel_args": [1024, 512, 256, 128], 
    "latent_channels": 16,            # Must match your VAE output or be handled by the UNet input layer
    "num_res_blocks_per_level": 3, 
    "time_dim": 512,                  # Dimension for sinusoidal time embeddings
}
```
To start the UNet training, ensure you have your latent_data prepared, and the scaling_factor calculated from the previous steps. 


```bash
torchrun --nproc_per_node=NUM_GPUS script/trainunet.py \
    --random_seed 42 \
    --lr 1e-4 \
    --scaling_factor <your_scaling_factor> \
    --latent_data_file ./preprocessed_data/latent_data.pkl \
    --batch_size 32 \
    --total_timesteps 1000 \
    --noise_scheduler_type "cosine" \   
    --epochs 100 \
    --metric_log_dir ./unet_logs \
    --ckpt_dir ./unet_checkpoints \
    --accum_grad_after_k_steps 2 \
    --val_after_k_optim_steps 32 \
    --save_ckpt_after_k_steps 5000

```

# :paintbrush: Inference and Image Generation
The script `script/generate_image.py`serves as the final stage of the pipeline: using the trained models to generate new images from pure noise. It combines the UNet (for iterative denoising) and the VAE Decoder (for mapping latents back to pixel space) using the DDIM (Denoising Diffusion Implicit Models) sampling algorithm.

To generate an image, you must provide the paths to both your VAE and UNet checkpoints, along with the scaling factor used during UNet training.

```
python script/generate_image.py \
    --vae_ckpt_path ./checkpoints/VAE_best_val.pth \
    --unet_ckpt_path ./unet_checkpoints/UNET_best_val.pth \
    --scaling_factor <your_scaling_factor> \
    --latent_shape 1 4 32 32 \
    --max_timesteps 1000 \
    --num_steps 50 \
    --noise_type "cosine" \
    --save_dir ./outputs \
    --image_name "generated_face"
```

### :rocket: Pre-trained weights
- VAE (256x256) : \[*pending*\]:hourglass:
- UNet : \[*pending*\]:hourglass:

These will be uploaded shortly to serve as a baseline for your own experiments. Feel free to star the repo to get notified when the checkpoints are ready.