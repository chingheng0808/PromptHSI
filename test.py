import torch
import numpy as np
import matplotlib.pyplot as plt
import clip
import os
from utils.dataset import PromptHSIDataset
from model import PromptHSI
from tqdm import tqdm

def test(args):
    img_size = (args.imgsize, args.imgsize)
    win_size = tuple(args.win_size)
    patch_size = tuple(args.pat_size)
    n_layers = tuple(args.n_layers)
    model = PromptHSI(
        img_size,
        in_channel=args.in_channel,
        embeding_dim=args.emb_dim,
        num_blocks_tf=args.n_tf,
        num_heads=args.n_head,
        num_layers=n_layers,
        window_size=win_size,
        patch_size=patch_size,
    )

    # Load the state dict
    state_dict = torch.load(args.ckpt)['model']
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()

    dataloader = torch.utils.data.DataLoader(
            PromptHSIDataset(args.root, img_size=224, long_prompt=False, interpolate=True, mode='test'),
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )

    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device="cpu")
        model_clip.eval()

    batch_idx = 0
    test_bar = tqdm(dataloader)

    restored_path = 'restored_results/test'
    if not os.path.exists(restored_path):
        os.makedirs(restored_path, exist_ok=True)

    print("Start generating testing results...")
    for step, (data) in enumerate(test_bar):
        with torch.no_grad():
            x, t, gt, fn = data["x"].permute(0, 3, 1, 2), data["desc"], data["gt"].permute(0, 3, 1, 2), data['fn']
            x = x.to('cuda').float()
            gt = gt.to('cuda').float()
            t_fea = torch.empty(x.shape[0], 1, 512).to('cuda')
            for i in range(x.shape[0]):
                vt_tok = clip.tokenize([t[i]])
                with torch.no_grad():
                    t_fea[i, :, :] = model_clip.encode_text(vt_tok).to('cuda')
            y, _, _, _, _, _, _, _, _ = model(x, gt, t_fea)
            
            y = y[0].cpu().detach().numpy() # (C, H, W)
            y = y.transpose(1,2,0) # (H, W, C)
            
            fn = fn[0].split('/')[-1].split('.')[0]
            np.save(os.path.join(restored_path, f'{fn}_restored.npy'), y)
    print("Finish generating testing results...")

if __name__ == "__main__":
    from options import options as args
    
    test(args)