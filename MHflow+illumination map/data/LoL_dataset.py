import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import pickle
import cv2
from torchvision.transforms import ToTensor
import random
import torchvision.transforms as T


# import pdb

class LoL_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.concat_histeq = all_opt["concat_histeq"] if "concat_histeq" in all_opt.keys() else False
        self.histeq_as_input = all_opt["histeq_as_input"] if "histeq_as_input" in all_opt.keys() else False
        self.log_low = opt["log_low"] if "log_low" in opt.keys() else False
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.use_noise = opt[
            'noise_prob'] if "noise_prob" in opt.keys() else False  # (opt['noise_prob'] and train) if "noise_prob" in opt.keys() else False
        self.noise_prob = opt['noise_prob'] if self.use_noise else None
        self.noise_level = opt['noise_level'] if "noise_level" in opt.keys() else 0
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        # pdb.set_trace()
        if train:
            self.root = os.path.join(self.root, 'our485')
        else:
            self.root = os.path.join(self.root, 'eval15')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):
        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'png' in x, low_list)
        pairs = []
        for idx, f_name in enumerate(low_list):
            pairs.append(
                [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  # [:, 4:-4, :],
                 cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB),
                 # [:, 4:-4, :],
                 f_name.split('.')[0]])
            # if idx > 10: break
            pairs[-1].append(self.hiseq_color_cv2_img(pairs[-1][0]))
        return pairs

    def hiseq_color_cv2_img(self, img):
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        return result

    def __getitem__(self, item):
        lr, hr, f_name, his = self.pairs[item]
        if self.histeq_as_input:
            lr = his

        if self.use_crop:
            hr, lr = random_crop(hr, lr, self.crop_size)

        if self.center_crop_hr_size:
            hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr,
                                                                                 self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        # hr = hr / 255.0
        # lr = lr / 255.0

        # if self.measures is None or np.random.random() < 0.05:
        #     if self.measures is None:
        #         self.measures = {}
        #     self.measures['hr_means'] = np.mean(hr)
        #     self.measures['hr_stds'] = np.std(hr)
        #     self.measures['lr_means'] = np.mean(lr)
        #     self.measures['lr_stds'] = np.std(lr)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        if self.use_noise and random.random() < self.noise_prob:
            lr = torch.randn(lr.shape) * (self.noise_level / 255) + lr
        if self.log_low:
            lr = torch.log(torch.clamp(lr + 1e-3, min=1e-3))
        # if self.gpu:
        #    hr = hr.cuda()
        #    lr = lr.cuda()
       # if self.concat_histeq:
        #    his = self.to_tensor(his)
         #   lr = torch.cat([lr, his], dim=0)

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name,}

class LoL_Dataset_v2(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.concat_histeq = all_opt["concat_histeq"] if "concat_histeq" in all_opt.keys() else False
        self.histeq_as_input = all_opt["histeq_as_input"] if "histeq_as_input" in all_opt.keys() else False
        self.log_low = opt["log_low"] if "log_low" in opt.keys() else False
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.use_resize = opt["use_resize"] if "use_resize" in opt.keys() else False
        self.use_noise = opt[
            'noise_prob'] if "noise_prob" in opt.keys() else False  # (opt['noise_prob'] and train) if "noise_prob" in opt.keys() else False
        self.noise_prob = opt['noise_prob'] if self.use_noise else None
        self.noise_level = opt['noise_level'] if "noise_level" in opt.keys() else 0
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        # pdb.set_trace()
        self.pairs = []
        self.train = train
        for sub_data in ['Synthetic']:  # ['Real_captured']: # :  ['Synthetic']:
            if train:
                root = os.path.join(self.root, sub_data, 'Train')
            else:
                root = os.path.join(self.root, sub_data, 'Test')
            self.pairs.extend(self.load_pairs(root))
        self.to_tensor = ToTensor()
        self.gamma_aug = opt['gamma_aug'] if 'gamma_aug' in opt.keys() else False

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):
        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = sorted(list(filter(lambda x: 'png' in x, low_list)))
        high_list = os.listdir(os.path.join(folder_path, 'high'))
        high_list = sorted(list(filter(lambda x: 'png' in x, high_list)))
        pairs = []
        for idx in range(len(low_list)):
            f_name_low = low_list[idx]
            f_name_high = high_list[idx]
            # if ('r113402d4t' in f_name_low or 'r17217693t' in f_name_low) or self.train: # 'r113402d4t' in f_name_low or 'r116825e2t' in f_name_low or 'r068812d7t' in f_name_low
            pairs.append(
                [os.path.join(folder_path, 'low', f_name_low),  # [:, 4:-4, :],
                 os.path.join(folder_path, 'high', f_name_high),  # [:, 4:-4, :],
                 f_name_high.split('.')[0]])
            # if idx > 10: break
        return pairs

    def __getitem__(self, item):
        lr_path, hr_path, f_name = self.pairs[item]
        lr = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)

        if self.use_crop:
            hr, lr = random_crop(hr, lr, self.crop_size)

        # if self.use_resize:
        #   hr, lr = img_resize(hr, lr, self.crop_size)

        if self.center_crop_hr_size:
            hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)
        dark = lr  # 输入暗图像
        R_split, G_split, B_split = torch.split(dark, 1, dim=0)
        zero_array = R_split * G_split * B_split
        zero_array[zero_array != 0] = 1
        zero_array = 1 - zero_array
        mask = zero_array
        return {'LQ': lr, 'GT': hr, 'zero_img': mask, 'LQ_path': f_name, 'GT_path': f_name}
    


class MIT_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.train = train

        self.concat_histeq = all_opt.get("concat_histeq", False)
        self.histeq_as_input = all_opt.get("histeq_as_input", False)
        self.log_low = opt.get("log_low", False)
        self.use_flip = opt.get("use_flip", False)
        self.use_rot = opt.get("use_rot", False)
        self.use_crop = opt.get("use_crop", False)
        self.use_resize = opt.get("use_resize", False)

        self.use_noise = opt.get("noise_prob", False)
        self.noise_prob = opt.get("noise_prob", None)
        self.noise_level = opt.get("noise_level", 0)

        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)

        self.to_tensor = ToTensor()
        self.gamma_aug = opt.get("gamma_aug", False)

        self.pairs = self.load_pairs(self.root)

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):
        low_list = sorted([f for f in os.listdir(os.path.join(folder_path, 'low')) if f.endswith('.png')])
        high_list = sorted([f for f in os.listdir(os.path.join(folder_path, 'high')) if f.endswith('.png')])
        pairs = []
        for low_img, high_img in zip(low_list, high_list):
            pairs.append([
                os.path.join(folder_path, 'low', low_img),
                os.path.join(folder_path, 'high', high_img),
                high_img.split('.')[0]
            ])
        return pairs
    
    def make_even(self,img, factor=2):
        h, w, _ = img.shape
        h_new = h - (h % factor)
        w_new = w - (w % factor)
        if h_new % 83 == 0:
            h_new = h_new - 1
        
        if w_new % 83 == 0:
            w_new = w_new - 1
        
        # print(f"Old dim ({h}, {w}),  New dim ({h_new}, {w_new})")
        return img[ :h_new,  :w_new]

    def __getitem__(self, idx):
        lr_path, hr_path, f_name = self.pairs[idx]
        lr = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)
        
        # Make sure dimensions are even
        # lr = self.make_even(lr, factor=2)
        # hr = self.make_even(hr, factor=2)

        if self.use_crop:
            hr, lr = random_crop(hr, lr, self.crop_size)

        if self.center_crop_hr_size:
            hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        # # 🛠 Fix for pixel_unshuffle: make dimensions divisible by 2
        # lr = self.make_even(lr, factor=2)
        # hr = self.make_even(hr, factor=2)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        R, G, B = torch.split(lr, 1, dim=0)
        zero_array = R * G * B
        zero_array[zero_array != 0] = 1
        mask = 1 - zero_array

        return {'LQ': lr, 'GT': hr, 'zero_img': mask, 'LQ_path': f_name, 'GT_path': f_name}



def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 1).copy()
    seg = seg if random_choice else np.flip(seg, 1).copy()
    return img, seg


def gamma_aug(img, gamma=0):
    max_val = img.max()
    img_after_norm = img / max_val
    img_after_norm = np.power(img_after_norm, gamma)
    return img_after_norm * max_val


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(0, 1)).copy()
    seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()
    return img, seg


# def random_crop(hr, lr, size_hr):
#     size_lr = size_hr

#     size_lr_x = lr.shape[0]
#     size_lr_y = lr.shape[1]

#     start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
#     start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

#     # LR Patch
#     lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :]

#     # HR Patch
#     start_x_hr = start_x_lr
#     start_y_hr = start_y_lr
#     hr_patch = hr[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]

#     return hr_patch, lr_patch


def random_crop(hr, lr, size_hr):
    # Ensure crop size is even (important for pixel_unshuffle)
    if size_hr % 2 != 0:
        size_hr -= 1  # Make it even

    size_lr = size_hr

    h, w, _ = lr.shape

    if h < size_lr or w < size_lr:
        raise ValueError(f"Image too small for cropping: got ({h}, {w}), required ({size_lr}, {size_lr})")

    start_x = np.random.randint(0, h - size_lr + 1)
    start_y = np.random.randint(0, w - size_lr + 1)

    # Apply the same crop to both LR and HR
    lr_patch = lr[start_x:start_x + size_lr, start_y:start_y + size_lr, :]
    hr_patch = hr[start_x:start_x + size_hr, start_y:start_y + size_hr, :]

    return hr_patch, lr_patch

def img_resize(hr, lr, size_hr):
    size_lr = size_hr

    # LR Patch
    lr_patch = cv2.resize(lr, (size_lr, size_lr))

    # HR Patch
    hr_patch = cv2.resize(hr, (size_hr, size_hr))

    return hr_patch, lr_patch


def center_crop(img, size):
    if img is None:
        return None
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[border:-border, border:-border, :]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]
