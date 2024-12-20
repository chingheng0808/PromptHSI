import numpy as np
import torch
import os

def normHSI(R):
    rmax, rmin = np.max(R), np.min(R)
    R = (R - rmin) / (rmax - rmin)
    return R

def interpolate_rows(image):
    h, w, c = image.shape
    for i in range(c):
        zero_rows = np.where(np.all(image[:, :, i] == 0, axis=1))[0]
        non_zero_rows = np.where(np.any(image[:, :, i] != 0, axis=1))[0]

        if len(zero_rows) == 0 or len(non_zero_rows) < 2:
            continue

        for zero_row in zero_rows:
            above_non_zero_row = (
                non_zero_rows[non_zero_rows < zero_row].max()
                if any(non_zero_rows < zero_row)
                else None
            )
            below_non_zero_row = (
                non_zero_rows[non_zero_rows > zero_row].min()
                if any(non_zero_rows > zero_row)
                else None
            )

            if above_non_zero_row is None and below_non_zero_row is None:
                continue
            elif above_non_zero_row is None:
                image[zero_row, :, i] = image[below_non_zero_row, :, i]
            elif below_non_zero_row is None:
                image[zero_row, :, i] = image[above_non_zero_row, :, i]
            else:
                weight = (zero_row - above_non_zero_row) / (
                    below_non_zero_row - above_non_zero_row
                )
                image[zero_row, :, i] = (1 - weight) * image[
                    above_non_zero_row, :, i
                ] + weight * image[below_non_zero_row, :, i]

    return image


def interpolate_channels(image):
    h, w, c = image.shape
    image_mean = np.mean(image, axis=(0, 1))
    zero_channels = np.where(image_mean == 0)[0]
    non_zero_channels = np.where(image_mean != 0)[0]

    if len(zero_channels) == 0 or len(non_zero_channels) < 2:
        return image

    for zero_channel in zero_channels:
        above_non_zero_row = (
            non_zero_channels[non_zero_channels < zero_channel].max()
            if any(non_zero_channels < zero_channel)
            else None
        )
        below_non_zero_row = (
            non_zero_channels[non_zero_channels > zero_channel].min()
            if any(non_zero_channels > zero_channel)
            else None
        )

        if above_non_zero_row is None and below_non_zero_row is None:
            continue
        elif above_non_zero_row is None:
            image[:, :, zero_channel] = image[:, :, below_non_zero_row]
        elif below_non_zero_row is None:
            image[:, :, zero_channel] = image[:, :, above_non_zero_row]
        else:
            weight = (zero_channel - above_non_zero_row) / (
                below_non_zero_row - above_non_zero_row
            )
            image[:, :, zero_channel] = (1 - weight) * image[
                :, :, above_non_zero_row
            ] + weight * image[:, :, below_non_zero_row]

    return image

def findMissingBands(hsi_deg):
    b, h = hsi_deg.shape[0], hsi_deg.shape[1]
    counts = 0
    first_missing_row = 0
    for i in range(b):
        for j in range(h):
            row = hsi_deg[i, j, :]
            all_zero = not np.any(row)
            if all_zero:
                first_missing_row = j
                break
        if all_zero:
            break
    for i in range(b):
        if not np.any(hsi_deg[i, first_missing_row, :]):
            counts += 1
    return counts

def longPrompt(description, hsi_deg):
    prompt = "This hyperspectral image"
    isBandmissing = False
    degradations = []

    if description == "":
        prompt = f"{prompt} does not face any degradation;"
    else:
        if "Missing" in description:
            isBandmissing = True
            nmb = findMissingBands(hsi_deg)
            if "Band-wise Missing" in description:
                prompt = f'{prompt}  {nmb} bands;'
            elif "Bands Complete Missing" in description:
                prompt = f'{prompt} faces with "band complete missing" on {nmb} bands;'
            elif "Partial Missing" in description:
                prompt = f'{prompt} faces with "bands partial missing" on {nmb} bands;'

        if "Thickly Cloudy" in description:
            degradations.append('"thickly cloudy"')
        elif "Thinly Cloudy" in description:
            degradations.append('"thinly cloudy"')

        if "Noisy" in description:
            degradations.append('"gaussian noise"')

        if isBandmissing:
            prompt = f'{prompt} it also confronts with {", ".join(degradations)};'
        else:
            prompt = f'{prompt} it confronts with {", ".join(degradations)};'

        if "Spatial Blurring" in description:
            if isBandmissing or degradations:
                prompt = f'{prompt} besides, there exists "blurring effect in spatial domain".'
            else:
                prompt = f'{prompt} there exists "blurring effect on spatial domain".'
        if "Spectral Blurring" in description:
            if isBandmissing or degradations:
                prompt = f'{prompt} besides, there exists "blurring effect in spectral domain".'
            else:
                prompt = f'{prompt} there exists "blurring effect in spectral domain".'
    return prompt

class PromptHSIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        img_size=224,
        long_prompt=False,
        mode="train",
        interpolate=False,
    ):
        """
        Synthetic degraded HSI dataset: training: 900, validation: 100, testing: 100
        Parameters:
        root (str): The root directory of the degraded dataset.
        img_size (int, optional): The size of the cropped image. Defaults to 224.
        long_prompt (bool, optional): Flag to indicate whether to use long prompts. Defaults to False.
        mode (str, optional): The mode of the dataset, either "train", "val", or "test". Defaults to "train".
        interpolate (bool, optional): Flag to indicate whether to interpolate data. Defaults to False.
        """
        super(PromptHSIDataset, self).__init__()

        self.root = root
        self.mode = mode
        self.interpolate = interpolate
        self.img_size = img_size
        self.long_prompt = long_prompt
        self.flist_deg, self.flist_gt = [], []

        files = os.listdir(os.path.join(root, "Degradation", mode))

        # both file lists with the same order
        for f in files:
            self.flist_deg.append(os.path.join(root, "Degradation", mode, f))
            self.flist_gt.append(
                os.path.join(
                    root,
                    mode,
                    f"{f[:-8]}.npy",
                )
            )

        self.len = len(self.flist_deg)

    def __getitem__(self, index):
        file = np.load(self.flist_deg[index])
        x = file["HSI"][:, : self.img_size, : self.img_size]
        t = np.array2string(file["Description"])  # convert to string from numpy array
        if self.long_prompt:
            t = longPrompt(t, x)
        t = t[1:-1]  # remove prifix and postfix " " of the string

        x = np.transpose(x, (1, 2, 0))  # (H, W, C)
        x = x.astype(np.float32)

        if self.interpolate:
            x = interpolate_rows(x)
            x = interpolate_channels(x)

        ## get the path of GT
        # load ground truth HSI and normalization
        gt = np.load(self.flist_gt[index])
        gt = gt.astype(np.float32)
        gt = normHSI(gt)[: self.img_size, : self.img_size, :]

        data = {"x": x, "desc": t, "gt": gt}

        if self.mode != "train":
            data["fn"] = self.flist_gt[index]
        return data  # degraded HSI after projected, degradation description, ground truth HSI

    def __len__(self):
        return self.len
