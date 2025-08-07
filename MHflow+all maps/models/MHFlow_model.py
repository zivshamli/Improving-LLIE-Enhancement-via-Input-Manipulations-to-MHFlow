import logging
from collections import OrderedDict
from utils.util import get_resume_paths, opt_get

import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from torch.cuda.amp import GradScaler, autocast
from models.modules.loss import CharbonnierLoss
import torch.nn.functional as F

logger = logging.getLogger('base')


class MHFlowModel(BaseModel):
    def __init__(self, opt, step):
        super(MHFlowModel, self).__init__(opt)
        self.opt = opt

        self.already_print_params_num = False

        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']
        self.hr_size = opt['datasets']['train']['GT_size']  # opt_get(opt, ['datasets', 'train', 'center_crop_hr_size'])
        # self.hr_size = 160 if self.hr_size is None else self.hr_size
        self.lr_size = self.hr_size // opt['scale']

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # self.max_grad_clip = 5
        # self.max_grad_norm = 100
        # L1 loss
        # self.cri_pix = CharbonnierLoss().to(self.device)
        # self.cri_pix = nn.L1Loss().to(self.device)

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)
        #
        # weight_l1 = opt_get(self.opt, ['train', 'weight_l1']) or 0
        # if weight_l1 and 1:
        #     missing_keys, unexpected_keys = self.netG.load_state_dict(torch.load(
        #         '/home/yufei/project/LowLightFlow/experiments/to_pretrain_netG/models/1000_G.pth'),
        #         strict=False)
        #     print('missing %d keys, unexpected %d keys' % (len(missing_keys), len(unexpected_keys)))
        # if self.device.type != 'cpu':
        if opt['gpu_ids'] is not None and len(opt['gpu_ids']) > 0:
            if opt['dist']:
                self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            elif len(opt['gpu_ids']) > 1:
                self.netG = DataParallel(self.netG, opt['gpu_ids'])
            else:
                self.netG.cuda()
        # print network
        # self.print_network()

        if opt_get(opt, ['path', 'resume_state'], 1) is not None:
            self.load()
        else:
            print("WARNING: skipping initial loading, due to resume_state None")

        if self.is_train:
            self.netG.train()

            self.init_optimizer_and_scheduler(train_opt)
            self.log_dict = OrderedDict()

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def init_optimizer_and_scheduler(self, train_opt):
        # optimizers
        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        if isinstance(wd_G, str): wd_G = eval(wd_G)
        optim_params_RRDB = []
        optim_params_other = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            # print(k, v.requires_grad)
            if v.requires_grad:
                if '.RRDB.' in k:
                    optim_params_RRDB.append(v)
                    # print('opt', k)
                else:
                    optim_params_other.append(v)
                # if self.rank <= 0:
                #     logger.warning('Params [{:s}] will not optimize.'.format(k))

        print('rrdb params', len(optim_params_RRDB))

        self.optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other, "lr": train_opt['lr_G'], 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
                {"params": optim_params_RRDB, "lr": train_opt.get('lr_RRDB', train_opt['lr_G']),
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': 1e-5}
            ]
        )

        self.scaler = GradScaler()

        self.optimizers.append(self.optimizer_G)
        # schedulers
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state'],
                                                     lr_steps_invese=train_opt.get('lr_steps_inverse', [])))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def add_optimizer_and_scheduler_RRDB(self, train_opt):
        # optimizers
        assert len(self.optimizers) == 1, self.optimizers
        assert len(self.optimizer_G.param_groups[1]['params']) == 0, self.optimizer_G.param_groups[1]
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    self.optimizer_G.param_groups[1]['params'].append(v)
        assert len(self.optimizer_G.param_groups[1]['params']) > 0

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        self.zero_channel = data['zero_img'].to(self.device) if 'zero_img' in data else torch.zeros_like(self.var_L)
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
    
    def get_module(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module
        else:
            return model

    def optimize_parameters(self, step):
        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        if train_RRDB_delay is not None and step > int(train_RRDB_delay * self.opt['train']['niter']) \
                and not self.get_module(self.netG).RRDB_training:
            if self.get_module(self.netG).set_rrdb_training(True):
                self.add_optimizer_and_scheduler_RRDB(self.opt['train'])

        # self.print_rrdb_state()

        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()
        # with autocast():
        losses = {}
        weight_fl = opt_get(self.opt, ['train', 'weight_fl'])
        weight_fl = 1 if weight_fl is None else weight_fl
        weight_l1 = opt_get(self.opt, ['train', 'weight_l1']) or 0
        flow_warm_up_iter = opt_get(self.opt, ['train', 'flow_warm_up_iter'])
        # print(step, flow_warm_up_iter)
        if flow_warm_up_iter is not None:
            if step > flow_warm_up_iter:
                weight_fl = 0
            else:
                weight_l1 = 0
        # print(weight_fl, weight_l1)
        if weight_fl > 0:
            if self.opt['optimize_all_z']:
                if self.opt['gpu_ids'] is not None and len(self.opt['gpu_ids']) > 0:
                    epses = [[] for _ in range(len(self.opt['gpu_ids']))]
                else:
                    epses = []
            else:
                epses = None
            # total_loss = 0

            z, nll, y_logits = self.netG(gt=self.real_H, lr=self.var_L, maps=calculate_guidance_maps(self.var_L), reverse=False,
                                         epses=epses,
                                         add_gt_noise=True)
            nll_loss = torch.mean(nll)
            losses['nll_loss'] = nll_loss * weight_fl

        if weight_l1 > 0:
            z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            sr, logdet = self.netG(lr=self.var_L, z=z.to(self.var_L.device), eps_std=0, maps=calculate_guidance_maps(self.var_L),
                                   reverse=True,
                                   add_gt_noise=True)
            if not torch.isnan(sr).any() and (not torch.isinf(sr).any()):
                l1_loss = self.cri_pix(sr, self.real_H)
                l1_loss = l1_loss * weight_l1
                if l1_loss <= 8.0:
                    losses['l1_loss'] = l1_loss
                else:
                    losses['l1_loss'] = torch.tensor(0)
            else:
                losses['l1_loss'] = torch.tensor(0)

        total_loss = sum(losses.values())
        # try:
        # total_loss.backward()
        # self.optimizer_G.step()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

        mean = total_loss.item()
        return mean

    # def gradient_clip(self):
    #     # gradient clip & norm, is not used in SRFlow
    #     if self.max_grad_clip is not None:
    #         torch.nn.utils.clip_grad_value_(self.netG.parameters(), self.max_grad_clip)
    #     if self.max_grad_norm is not None:
    #         torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.max_grad_norm)

    def print_rrdb_state(self):
        for name, param in self.get_module(self.netG).named_parameters():
            if "RRDB.conv_first.weight" in name:
                print(name, param.requires_grad, param.data.abs().sum())
        print('params', [len(p['params']) for p in self.optimizer_G.param_groups])

    def test(self):
        self.netG.eval()
        self.fake_H = {}
        if self.heats is not None:
            for heat in self.heats:
                for i in range(self.n_sample):
                    z = self.get_z(heat, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                    with torch.no_grad():
                        self.fake_H[(heat, i)], logdet = self.netG(lr=self.var_L, z=z,
                                                                   eps_std=heat, reverse=True, maps=calculate_guidance_maps(self.var_L),
                                                                   add_gt_noise=True)
        else:
            z = self.get_z(0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            with torch.no_grad():
                # torch.cuda.reset_peak_memory_stats()
                self.fake_H[(0, 0)], logdet = self.netG(lr=self.var_L, z=z.to(self.var_L.device),
                                                        eps_std=0, reverse=True,
                                                        maps=calculate_guidance_maps(self.var_L),
                                                        add_gt_noise=True)
        self.netG.train()
        return None

    def get_encode_nll(self, lq, gt):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, maps=calculate_guidance_maps(lq))
        self.netG.train()
        return nll.mean().item()

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses)[0]

    def get_encode_z(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise, maps=calculate_guidance_maps(lq))
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise, maps=calculate_guidance_maps(lq))
        self.netG.train()
        return z, nll

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()
        if heat is None:
            heat = 0
        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z.cuda(), eps_std=heat, reverse=True, add_gt_noise=True,maps=calculate_guidance_maps(lq))
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed: torch.manual_seed(seed)
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
            C = self.get_module(self.netG).flowUpsamplerNet.C
            H = int(self.opt['scale'] * lr_shape[2] // self.get_module(self.netG).flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.get_module(self.netG).flowUpsamplerNet.scaleW)
            z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W)) if heat > 0 else torch.zeros(
                (batch_size, C, H, W))
        else:
            L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
            fac = 2 ** L
            H = int(self.opt['scale'] * lr_shape[2] // self.get_module(self.netG).flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.get_module(self.netG).flowUpsamplerNet.scaleW)
            size = (batch_size, 3 * fac * fac, H, W)
            z = torch.normal(mean=0, std=heat, size=size) if heat > 0 else torch.zeros(size)
        return z

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        if self.heats is not None:
            for heat in self.heats:
                for i in range(self.n_sample):
                    out_dict[('NORMAL', heat, i)] = self.fake_H[(heat, i)].detach()[0].float().cpu()
        else:
            out_dict['NORMAL'] = self.fake_H[(0, 0)].detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            return

        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'RRDB'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=load_submodule)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

def calculate_upm(batch_rgb):
    R, G, B = batch_rgb[:, 0:1], batch_rgb[:, 1:2], batch_rgb[:, 2:3]
    denom = R + G + B + 1e-4
    rg = R / denom
    rb = G / denom
    gb = B / denom

    threshold = 0.1
    mask = (rg > threshold) & (rb > threshold) & (gb > threshold)
    return mask.float()  # shape: (B, 1, H, W)

def calculate_batch_illum_map(var_L):
    if var_L.dim() == 4:
        batch_size = var_L.shape[0]
    elif var_L.dim() == 3:
        var_L = var_L.unsqueeze(0)  # make it a batch of 1
        batch_size = 1
    else:
        raise TypeError(f"Expected torch.Tensor of dimension 3 or 4, got {type(var_L)} of dim {var_L.dim()} and shape {var_L.shape}")

    batch_illum_map = []
    for i in range(batch_size):
        illum_map = estimate_illumination_map(var_L[i])
        batch_illum_map.append(illum_map)

    return torch.stack(batch_illum_map)

def estimate_illumination_map(image: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    """
    Estimate an illumination map as the maximum channel value per pixel,
    followed by a box filter for refinement.

    Args:
        image (torch.Tensor): Low-light image tensor of shape (3, H, W) or (B, 3, H, W)
        kernel_size (int): Size of the box filter

    Returns:
        torch.Tensor: Illumination map of shape (H, W) or (B, 1, H, W)
    """
    if image.dim() == 4:
        B, C, H, W = image.shape
        max_rgb = torch.max(image, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        illum_map = torch.nn.functional.avg_pool2d(max_rgb, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return illum_map
    elif image.dim() == 3:
        C, H, W = image.shape
        max_rgb = torch.max(image, dim=0, keepdim=True)[0].unsqueeze(0)  # (1, 1, H, W)
        illum_map = torch.nn.functional.avg_pool2d(max_rgb, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return illum_map.squeeze(0)  # (1, H, W)
    else:
        raise ValueError("Expected input of shape (3, H, W) or (B, 3, H, W)")
    
def calculate_edge_map(batch_rgb):
    # Convert to grayscale: (B, 3, H, W) → (B, 1, H, W)
    gray = 0.299 * batch_rgb[:, 0:1] + 0.587 * batch_rgb[:, 1:2] + 0.114 * batch_rgb[:, 2:3]

    # Simple Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)

    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)

    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    edge = edge / (edge.max() + 1e-5)  # normalize
    return edge  # shape: (B, 1, H, W)


def calculate_guidance_maps(var_L: torch.Tensor) -> torch.Tensor:
    """
    Calculate a tensor of 3 guidance maps per image in the batch:
    - UPM (Unbalanced Point Map)
    - Illumination Map
    - Edge Map

    Args:
        var_L (torch.Tensor): Tensor of shape (B, 3, H, W)

    Returns:
        torch.Tensor: Tensor of shape (B, 3, H, W)
    """
    assert var_L.dim() == 4 and var_L.size(1) == 3, "Input must be a (B, 3, H, W) tensor"

    upm = calculate_upm(var_L)                    # (B, 1, H, W)
    illum = calculate_batch_illum_map(var_L)      # (B, 1, H, W)
    edge = calculate_edge_map(var_L)              # (B, 1, H, W)

    return torch.cat([upm, illum, edge], dim=1)   # (B, 3, H, W)