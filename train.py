import os
import time
import numpy as np
from tqdm import tqdm
from model import PromptHSI
import torch
import torch.optim as optim
from utils.dataset import PromptHSIDataset
import math
from tensorboardX import SummaryWriter
import clip
import torchvision.transforms.functional as TF
import random
from torchvision import transforms
from utils.metrics import psnr, sam, rmse, ergas

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def plog(msg, savename):
    print(msg)
    with open(
        f"log/mylog-{savename}.txt",
        "a",
    ) as fp:
        fp.write(msg + "\n")

def tuple2str(t):
    return "-".join([str(i) for i in t])

def randCropParams(x, size=112):
    i, j, h, w = transforms.RandomCrop.get_params(x, output_size=(size, size))
    # Should random horizontal flipping?
    hf = False
    vf = False
    if random.random() > 0.5:
        hf = True
    # Should random vertical flipping?
    if random.random() > 0.5:
        vf = True
    return i, j, h, w, hf, vf

def cropImg(x, i, j, h, w, hf, vf, device='cuda'):
    x = TF.crop(x, i, j, h, w)
    if hf:
        x = TF.hflip(x)
    # Random vertical flipping
    if vf:
        x = TF.vflip(x)
    return x.to(device)

def trainer(args, only_val=False):
    ## Reading files
    print(f'Interpolation: {args.intp}, Long Prompt: {args.long}')
    device = 'cuda'
    train_loader = torch.utils.data.DataLoader(
        PromptHSIDataset(args.root, img_size=args.imgsize, long_prompt=args.long, mode="train", interpolate=args.intp),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        PromptHSIDataset(args.root, img_size=args.imgsize, long_prompt=args.long, mode="val", interpolate=args.intp),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory = True
    )
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)
        model_clip.eval()

    savename = args.savename
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
    model.cuda()

    state_dict = None
    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if args.state_dict is not None:
        state_dict = torch.load(args.state_dict)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(torch.load(state_dict["optimizer"]))
    model.train()
    
    L1Loss = torch.nn.L1Loss()
    writer = SummaryWriter(
        f"log/tensorboard-{savename}"
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5, last_epoch=-1, verbose=False)

    if not os.path.isdir("ckpt"):
        os.mkdir("ckpt")
    if not os.path.isdir("log"):
        os.mkdir("log")

    resume_ind = 0 if state_dict is None else state_dict["epoch"]
    step = resume_ind
    best_sam = math.inf if state_dict is None else state_dict["sam"]

    for epoch in range(resume_ind + 1, args.epochs + 1):
        running_loss, running_loss1, running_loss2, running_loss3, running_loss4 = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        loss = 0.0
        # Initialize the tqdm progress bar
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{args.epochs}/{loss}",
            unit="batch",
        )

        if not only_val:
            for batch_idx, (data) in pbar:
                x, t, gt = data["x"], data["desc"], data["gt"]
                # print(t)
                optimizer.zero_grad()
                i, j, w, h, hf, vf = randCropParams(
                    x.permute(0, 3, 1, 2), size=args.cropsize
                )
                x = cropImg(
                    x.permute(0, 3, 1, 2), i, j, w, h, hf, vf, device=device
                ).float()
                gt = cropImg(
                    gt.permute(0, 3, 1, 2), i, j, w, h, hf, vf, device=device
                ).float()

                t_fea = torch.empty(x.shape[0], 1, 512).to(device)
                for i in range(x.shape[0]):
                    t_tok = clip.tokenize([t[i]]).to(device)
                    with torch.no_grad():
                        t_fea[i, :, :] = model_clip.encode_text(t_tok).to(device)
                
                _, _, _, _, _, loss1, loss2, loss3, loss4 = model(x, gt, t_fea)
                loss1 = loss1.sum()
                loss2 = loss2.sum()
                loss3 = loss3.sum()
                loss4 = loss4.sum()
                loss = loss1 + 0.01 * loss2 + 0.001 * loss3 + 0.01 * loss4
                loss.backward()
                # print(
                #     f"{loss.item()}, L1: {loss1.item()}, SAM: {loss3.item()}, MSE: {loss2.item()}, SWT: {loss4.item()}"
                # )

                optimizer.step()
                running_loss += loss.item()
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()
                running_loss4 += loss4.item()

        
        print("Start Validation")
        with torch.no_grad():
            rmses, sams, l1loss, fnames, psnrs, ergass = [], [], [], [], [], []
            start_time = time.time()
            for ind2, (data) in enumerate(val_loader):
                vx, vt, vgt, vfn = data["x"], data["desc"], data["gt"], data["fn"]
                model.eval()
                vx = vx.to(device).permute(0, 3, 1, 2).float()
                vgt = vgt.to(device).permute(0, 3, 1, 2).float()
                vt_fea = torch.empty(vx.shape[0], 1, 512).to(device)
                for i in range(vx.shape[0]):
                    vt_tok = clip.tokenize([vt[i]]).to(device)
                    with torch.no_grad():
                        vt_fea[i, :, :] = model_clip.encode_text(vt_tok).to(device)
                val_dec, _, _, _, _, _, _, _, _ = model(vx, vgt, vt_fea)

                ## Recovery to image HSI
                val_batch_size = len(vfn)

                for bt in range(val_batch_size):
                    constructed_hsi = val_dec[bt]
                    GT = vgt[bt]
                    l1loss.append(L1Loss(constructed_hsi, GT).item())
                    constructed_hsi = constructed_hsi.cpu().detach().numpy()
                    GT = GT.cpu().detach().numpy()

                    sams.append(sam(constructed_hsi, GT))
                    psnrs.append(psnr(constructed_hsi, GT))   
                    rmses.append(rmse(constructed_hsi, GT))
                    ergass.append(ergas(constructed_hsi, GT))

            ep = time.time() - start_time
            ep = ep / len(sams)
            plog(
                "[epoch: %d, batch: %5d] Total-Loss: %.3f, L1loss: %.3f, bandMSE: %.3f, sam-loss: %.3f, swt-loss: %.3f, val-L1Loss: %.3f, val-RMSE: %.3f, val-ERGAS: %.3f, val-SAM: %.3f, val-PSNR: %.3f, AVG-Time: %.3f, LR: %f"
                % (
                    epoch,
                    batch_idx + resume_ind + 1,
                    loss.item(),
                    loss1.item(),
                    loss2.item(),
                    loss3.item(),
                    loss4.item(),
                    np.mean(l1loss),
                    np.mean(rmses),
                    np.mean(ergass),
                    np.mean(sams),
                    np.mean(psnrs),
                    ep,
                    scheduler.get_last_lr()[0],
                ), savename
            )
            ## Dump the SAM/RMSE/PSNR for each image
            writer.add_scalar("Validation/RMSE", np.mean(rmses), step)
            writer.add_scalar("Validation/ERGAS", np.mean(ergass), step)
            writer.add_scalar("Validation/SAM", np.mean(sams), step)
            writer.add_scalar("Validation/PSNR", np.mean(psnrs), step)

            writer.add_scalar("Training/L1Loss", running_loss1, step)
            writer.add_scalar("Training/BandWiseLoss", running_loss2, step)
            writer.add_scalar("Training/SAMLoss", running_loss3, step)
            writer.add_scalar("Training/SWTLoss", running_loss4, step)
            writer.add_scalar("Training/Total running loss", running_loss, step)

            if best_sam > np.mean(sams):
                best_sam = np.mean(sams)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "sam": np.mean(sams),
                        "psnr": np.mean(psnrs),
                        "rmse": np.mean(rmses),
                        "ergas": np.mean(ergass),
                        "epoch": epoch,
                        "lr": scheduler.get_last_lr()[0],
                        'optimizer': optimizer.state_dict(),
                    },
                    f"ckpt/BEST-{savename}.pth",
                )

        if (epoch) % 50 == 0 and epoch >= 1:
            torch.save(
                {
                    "model": model.state_dict(),
                    "sam": np.mean(sams),
                    "psnr": np.mean(psnrs),
                    "rmse": np.mean(rmses),
                    "ergas": np.mean(ergass),
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    'optimizer': optimizer.state_dict(),
                },
                f"ckpt/EP{epoch}-{savename}.pth",
            )

        model.train()
        scheduler.step()
        step += 1
    
    ################ Test #################
    test_loader = torch.utils.data.DataLoader(PromptHSIDataset(
            args.root, img_size=args.imgsize, long_prompt=args.long, mode="test", interpolate=args.intp
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    state_dict = torch.load(f"ckpt/BEST-{savename}.pth")['model']

    # Load the modified state_dict into the model
    model.load_state_dict(state_dict)
    model.to('cuda')
    model.eval()
        
    rmses, sams, psnrs, ergass = [], [], [], []
        
    for data in test_loader:
        x, t, gt = data["x"].permute(0, 3, 1, 2), data["desc"], data["gt"].permute(0, 3, 1, 2)
        x = x.to('cuda').float()
        gt = gt.to('cuda').float()
        t_fea = torch.empty(x.shape[0], 1, 512).to('cuda')
        for i in range(x.shape[0]):
            vt_tok = clip.tokenize([t[i]]).to('cuda')
            with torch.no_grad():
                t_fea[i, :, :] = model_clip.encode_text(vt_tok).to('cuda')
        y, _, _, _, _, _, _, _, _ = model(x, gt, t_fea)
        
        gt = gt[0].cpu().detach().numpy()
        y = y[0].cpu().detach().numpy()
        sams.append(sam(y, gt))
        psnrs.append(psnr(y, gt))
        rmses.append(rmse(y, gt))
        ergass.append(ergas(y, gt))
    plog(
        "\n[Testing:]\n test-PSNR: %.3f, test-SAM: %.3f, test-RMSE: %.3f, test-ERGAS: %.3f"
        % (
            np.mean(psnrs),
            np.mean(sams),
            np.mean(rmses),
            np.mean(ergass),
        ), savename
    )

if __name__ == "__main__":
    from options import options as args
    trainer(args)
