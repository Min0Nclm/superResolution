import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import sys
from datetime import datetime
from torchvision.transforms import ToTensor, Resize, InterpolationMode, ToPILImage

from models import LFPN

class Logger(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class SuperResolutionDataset(Dataset):

    def __init__(self, hr_image_dir, scale_factor, patch_size=96, is_train=True):
        super(SuperResolutionDataset, self).__init__()
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.is_train = is_train
        

        self.hr_image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            self.hr_image_paths.extend(glob.glob(os.path.join(hr_image_dir, ext)))
        self.hr_image_paths = sorted(self.hr_image_paths)

        self.hr_transform = ToTensor()
        self.lr_transform = Resize((patch_size // scale_factor, patch_size // scale_factor), interpolation=InterpolationMode.BICUBIC)

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_image_paths[index]).convert("RGB")
        
        if self.is_train:

            w, h = hr_image.size
            if w < self.patch_size or h < self.patch_size:
                hr_image = hr_image.resize((max(w, self.patch_size), max(h, self.patch_size)), Image.BICUBIC)
            
            w, h = hr_image.size
            rand_w = torch.randint(0, w - self.patch_size + 1, (1,)).item()
            rand_h = torch.randint(0, h - self.patch_size + 1, (1,)).item()
            hr_patch = hr_image.crop((rand_w, rand_h, rand_w + self.patch_size, rand_h + self.patch_size))
            hr_tensor = self.hr_transform(hr_patch)
        else:

            w, h = hr_image.size
            sw, sh = w // self.scale_factor, h // self.scale_factor
            lr_w, lr_h = sw * self.scale_factor, sh * self.scale_factor
            hr_patch = hr_image.crop(( (w - lr_w) // 2, (h - lr_h) // 2, (w + lr_w) // 2, (h + lr_h) // 2 ))
            hr_tensor = self.hr_transform(hr_patch)

        if self.is_train:
            lr_pil = ToPILImage()(hr_tensor)
            lr_pil = self.lr_transform(lr_pil)
            lr_tensor = self.hr_transform(lr_pil)
        else:
            hr_pil = ToPILImage()(hr_tensor)
            w, h = hr_pil.size
            sw, sh = w // self.scale_factor, h // self.scale_factor
            lr_transform = Resize((sh, sw), interpolation=InterpolationMode.BICUBIC)
            lr_pil = lr_transform(hr_pil)
            lr_tensor = self.hr_transform(lr_pil)

        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.hr_image_paths)

def calculate_psnr(sr, hr, rgb_range=1.0):
    if hr.nelement() == 1: return 0
    diff = sr - hr
    mse = diff.pow(2).mean()
    if mse == 0:
        return float('inf')
    return -10 * torch.log10(mse)

if __name__ == '__main__':
    SCALE_FACTOR = 2
    MAX_ITERATIONS = 1000000
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-4
    TRAIN_HR_DIR = "data/Train/HR"
    VALID_HR_DIR = "data/Validation/HR"
    RESULTS_DIR = "results"

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(RESULTS_DIR, run_timestamp)
    os.makedirs(output_dir, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(output_dir, "lfpn_x2_best.pt")
    LOG_FILE_PATH = os.path.join(output_dir, "log.txt")


    sys.stdout = Logger(LOG_FILE_PATH)
    print(f"Results save in: {LOG_FILE_PATH}")

    if not os.path.exists(TRAIN_HR_DIR) or not os.path.exists(VALID_HR_DIR):
        print(
            f"Error: Please make sure the dataset folders exist and contain images:\n- Training set: {TRAIN_HR_DIR}\n- Validation set: {VALID_HR_DIR}")
        os.makedirs(TRAIN_HR_DIR, exist_ok=True)
        os.makedirs(VALID_HR_DIR, exist_ok=True)
        print("The folders have been created automatically. Please put the dataset inside and rerun the program.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = LFPN(scale=SCALE_FACTOR).to(device)
    print("Init sucess")

    train_dataset = SuperResolutionDataset(hr_image_dir=TRAIN_HR_DIR, scale_factor=SCALE_FACTOR, is_train=True)
    valid_dataset = SuperResolutionDataset(hr_image_dir=VALID_HR_DIR, scale_factor=SCALE_FACTOR, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    print(f"loadDataset: {len(train_dataset)}train image, {len(valid_dataset)}valid image")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERATIONS, eta_min=1e-7)

    best_psnr = 0.0
    current_iter = 0
    epoch = 0
    print("\n--- Start ---")

    while current_iter < MAX_ITERATIONS:
        model.train()
        epoch_loss = 0
        epoch_start_iter = current_iter
        
        for i, (lr_batch, hr_batch) in enumerate(train_loader):
            if current_iter >= MAX_ITERATIONS:
                break

            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            current_iter += 1

            if current_iter % 100 == 0:
                print(f"\rEpoch [{epoch+1}], Iter [{current_iter}/{MAX_ITERATIONS}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.1e}", end='')
        
        epoch += 1
        print(f"\nEpoch [{epoch}] finish. Average Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        avg_psnr = 0
        with torch.no_grad():
            for lr_batch, hr_batch in valid_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                sr_batch = model(lr_batch)
                avg_psnr += calculate_psnr(sr_batch, hr_batch)
        
        avg_psnr = avg_psnr / len(valid_loader)
        print(f"-- Validation finish. Average PSNR: {avg_psnr:.2f} dB --")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            print(f"** New best PSNR: {best_psnr:.2f} dB. save {MODEL_SAVE_PATH} **")
            scripted_model = torch.jit.script(model)
            scripted_model.save(MODEL_SAVE_PATH)
        print("-" * 30)

    print(f"\n--- End of {MAX_ITERATIONS} iterations ---")
