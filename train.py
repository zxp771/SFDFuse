# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net_fft_detail_01201 import Restormer_Encoder, Restormer_Decoder_Phase,DetailFeatureExtraction,BaseFeatureExtraction_Pool,FeatureInteractionBlock
from utils.dataset import H5Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss_fft import Fusionloss, cc, waveletloss,L_TV
import kornia
import matplotlib.pyplot as plt

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
criteria_fusion = Fusionloss()
model_str = 'CDDFuse'

# . Set the hyper-parameters for training
num_epochs = 90 # total epoch
epoch_gap = 30 # epoches of Phase I

lr = 1e-4
weight_decay = 0
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1.  # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.  # alpha2 and alpha4 2
coeff_tv = 5#5

coeff_wavelets = 1.

#coeff_TV =1

clip_grad_norm_value = 0.01
optim_step = 10
optim_gamma = 0.5

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder_Phase = nn.DataParallel(Restormer_Decoder_Phase()).to(device)
#BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
#BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction_Pool(dim=64)).to(device)
BaseFuseLayer = nn.DataParallel(FeatureInteractionBlock(dim=64, num_heads=8,init_fusion=True)).to(device)
#DetailFuseLayer = nn.DataParallel(DEABlockTrain(conv=default_conv,dim=64,kernel_size=3)).to(device)
#DetailFuseLayer = nn.DataParallel(WTConv2d(in_channels=64, out_channels=64, kernel_size=3, wt_levels=2)).to(device)
#DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
DetailFuseLayer = nn.DataParallel(FeatureInteractionBlock(dim=64, num_heads=8,init_fusion=False)).to(device)
#DetailFuseLayer = nn.DataParallel(FeatureInteractionBlock(dim=64, num_heads=8)).to(device)

#FuseLayer = nn.DataParallel(FeatureInteractionBlock(dim=64, num_heads=8)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder_Phase.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
   DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer3 = torch.optim.Adam(
#     FuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

# scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=optim_step, eta_min=1e-6)
# scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=optim_step, eta_min=1e-6)
# scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=optim_step, eta_min=1e-6)
# scheduler4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer4, T_max=optim_step, eta_min=1e-6)


MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
Wavelet_Loss = waveletloss()
L_TV = L_TV()

# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
draw_loss = []

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder_Phase.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()
        # FuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder_Phase.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()
        # FuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < epoch_gap:  # Phase I

            feature_VIS_ll, feature_VIS_detail = DIDF_Encoder(data_VIS)
            feature_I_ll, feature_I_detail = DIDF_Encoder(data_IR)

            data_VIS_hat, feature_vis_hat = DIDF_Decoder_Phase(data_VIS, feature_VIS_ll, feature_VIS_detail)
            data_IR_hat, feature_ir_hat = DIDF_Decoder_Phase(data_IR, feature_I_ll, feature_VIS_detail)

            cc_loss_B = cc(feature_VIS_ll, feature_I_ll)
            cc_loss_D = cc(feature_VIS_detail, feature_I_detail)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)#5 ssim
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

            #tv_loss = (L_TV(feature_vis_hat)+L_TV(feature_ir_hat))/2

            #Wavelet_loss = Wavelet_Loss(a0,a1,s0,s1)
            # for name, parameters in Wavelet_Loss.named_parameters():
            #     print(name, ':', parameters, parameters.size())
            # print(list(Wavelet_Loss.parameters()))

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss# + coeff_TV * tv_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder_Phase.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
        else:  # Phase II
            feature_VIS_ll, feature_VIS_detail = DIDF_Encoder(data_VIS)
            feature_I_ll, feature_I_detail = DIDF_Encoder(data_IR)

            #feature_F_ll = BaseFuseLayer(feature_VIS_ll + feature_I_ll)
            #feature_F_detail = DetailFuseLayer(feature_VIS_detail + feature_I_detail)
            feature_F_ll = BaseFuseLayer(feature_VIS_ll,feature_I_ll)
            feature_F_detail = DetailFuseLayer(feature_VIS_detail,feature_I_detail)
            #feature_F_detail = DetailFuseLayer(feature_VIS_detail, feature_I_detail)



            data_Fuse, feature_F = DIDF_Decoder_Phase(data_VIS, feature_F_ll, feature_F_detail)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)#5*ssim
            mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)

            cc_loss_B = cc(feature_VIS_ll, feature_I_ll)
            cc_loss_D = cc(feature_VIS_detail, feature_I_detail)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            #Wavelet_loss = Wavelet_Loss(a0,a1,s0,s1)
            #tv_loss = L_TV(feature_F)
            fusionloss, _, _= criteria_fusion(data_VIS, data_IR, data_Fuse)

            loss = fusionloss + coeff_decomp * loss_decomp# + coeff_TV * tv_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder_Phase.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
               DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            # nn.utils.clip_grad_norm_(
            #     FuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()


        # torch.cuda.empty_cache()
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [mse_loss_V:%f] [mse_loss_I:%f] [cc_loss_B:%f] [cc_loss_D:%f] [loss_decomp:%f] [Gradient_loss:%f]ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                mse_loss_V.item(),
                mse_loss_I.item(),
                cc_loss_B.item(),
                cc_loss_D.item(),
                loss_decomp.item(),
                Gradient_loss.item(),
                #tv_loss.item(),[tv_loss:%f]
                #Wavelet_loss.item(),
                time_left,
            )
        )
        draw_loss.append(loss.item())
        #draw_decomp.append(loss_decomp.item())

    # adjust the learning rate
    scheduler1.step()
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
       optimizer4.param_groups[0]['lr'] = 1e-6

    # torch.cuda.empty_cache()

if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder_Phase': DIDF_Decoder_Phase.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/CDDFuse_" + timestamp + '.pth'))
