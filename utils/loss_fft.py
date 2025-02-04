import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia



class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)#l1_loss
        Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        loss_ssim = Loss_ssim(x_in_max, generate_img)
        #loss_mse = F.mse_loss(x_in_max, generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)#l1_loss
        # Smoothness Regularization for Gradient Loss
        #smooth_loss = smoothness_regularizer(generate_img)

        loss_total=loss_in+10*loss_grad+loss_ssim#+0.01*smooth_loss loss_in
        return loss_total,loss_in,loss_grad#,loss_ssim#,smooth_loss

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


def improved_cc(img1, img2):
    eps = torch.finfo(torch.float32).eps

    # 形状重塑
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)

    # 均值和标准差标准化
    img1_mean = img1.mean(dim=-1, keepdim=True)
    img1_std = img1.std(dim=-1, keepdim=True) + eps
    img2_mean = img2.mean(dim=-1, keepdim=True)
    img2_std = img2.std(dim=-1, keepdim=True) + eps

    img1_normalized = (img1 - img1_mean) / img1_std
    img2_normalized = (img2 - img2_mean) / img2_std

    # 使用余弦相似度计算相关性
    cc = torch.sum(img1_normalized * img2_normalized, dim=-1)
    cc = torch.clamp(cc, -1., 1.)  # 限制范围
    return cc.mean()


class waveletloss(nn.Module):
    def __init__(self):
        super(waveletloss, self).__init__()

    def forward(self,a0,a1,s0,s1):
        inner_product = torch.sum((-1) ** torch.arange(s0.size(1)).float().cuda() * torch.matmul(a0.transpose(2,3),s0).squeeze())+ \
        torch.sum((-1) ** torch.arange(s0.size(1)).float().cuda() * torch.matmul(a1.transpose(2,3),s1).squeeze())
        L_wavelet = (inner_product ** 2)/(s0.size(0) * s0.size(1))
        return L_wavelet

def smoothness_regularizer(img):
    dy, dx = torch.abs(img[:, :, :-1] - img[:, :, 1:])+ 1e-8, torch.abs(img[:, :-1, :] - img[:, 1:, :])+ 1e-8
    return torch.mean(dy) + torch.mean(dx)
