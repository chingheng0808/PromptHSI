import os
from utils.dataset import interpolate_channels, interpolate_rows, normHSI
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import PromptHSI
import clip
import argparse

def demo_infer(args):
    vis_channel = [int(args.R), int(args.G), int(args.B)]
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device="cpu")
        model_clip.eval()
    
    ## This hsi encouter "Thinly Cloudy, Noisy, Spectral Blurring, Band-wise Missing" 
    
    hsi_deg = np.load(args.target_path)
    hsi_deg = hsi_deg[:, : args.img_size, : args.img_size]
    hsi_deg = np.transpose(hsi_deg, (1, 2, 0))
    hsi_deg = hsi_deg.astype(np.float32)
    max_, min_ = np.max(hsi_deg[:, :, vis_channel]), np.min(hsi_deg[:, :, vis_channel])
    plt.imsave('demo/hsi_deg.png', (hsi_deg[:, :, vis_channel] - min_) / (max_ - min_))
    t = args.prompt
    
    if "Missing" in t or "missing" in t:
        hsi_deg = interpolate_rows(hsi_deg)
        hsi_deg = interpolate_channels(hsi_deg)
        # plt.imsave('demo/hsi_deg_intp.png', (hsi_deg[:, :, vis_channel] - min_) / (max_ - min_))
    
    model = PromptHSI(
            (224,224),
            in_channel=172,
            embeding_dim=64,
            num_blocks_tf=2,
            num_heads=8,
            num_layers=(2,1),
            window_size=(7,7,7),
            patch_size=(4,4,4),
        )
    state_dict = torch.load(args.ckpt)['model']
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()  
    
    hsi_deg = torch.tensor(hsi_deg.transpose(2, 0, 1)).unsqueeze(0).to('cuda').float()
    t_fea = torch.empty(1, 1, 512).to('cuda')
    vt_tok = clip.tokenize([t])
    with torch.no_grad():
        t_fea[0, :, :] = model_clip.encode_text(vt_tok).to('cuda')
    
    with torch.no_grad():
        restored, _,_,_,_,_,_,_,_ = model(hsi_deg, torch.randn(1, 172, args.img_size, args.img_size).to('cuda'), t_fea)
        np.save('demo/restored_hsi.npy', restored.cpu().detach().numpy().astype(np.float16))
        
        restored_rgb = restored.squeeze(0).permute(1,2,0).cpu().detach().numpy()[:, :, [args.R, args.G, args.B]]
        plt.imsave('demo/restored_rgb.png', (restored_rgb - min_) / (max_ - min_))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='ckpt/pretrained_prompthsi.pth')
    parser.add_argument("--target_path", type=str, default="demo/degraded_hsi.npy")
    parser.add_argument("--prompt", type=str, default="Thickly Cloudy, Spectral Blurring, Noisy, Band-wise Missing")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--R", type=int, default=24)
    parser.add_argument("--G", type=int, default=14)
    parser.add_argument("--B", type=int, default=5)
    
    args = parser.parse_args()
    
    demo_infer(args)