from net import Encoder, Decoder,FeatureInteractionBlock
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import cv2
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
ckpt_path=r"models/SFDFuse_01-31-02-13.pth"
for dataset_name in ["MSRS","TNO","RoadScene","M3FD"]:
    print("\n"*2+"="*80)
    model_name="SFDFuse    "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name)
    test_out_folder=os.path.join('test_result',dataset_name)#+'out')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Encoder()).to(device)
    Decoder = nn.DataParallel(Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(FeatureInteractionBlock(dim=64,num_heads=8,init_fusion=True)).to(device)
    DetailFuseLayer = nn.DataParallel(FeatureInteractionBlock(dim=64, num_heads=8,init_fusion=False)).to(device)


    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])

    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):

            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_VIS_ll, feature_VIS_detail = Encoder(data_VIS)
            feature_I_ll, feature_I_detail = Encoder(data_IR)
            feature_F_ll = BaseFuseLayer(feature_VIS_ll,feature_I_ll)
            feature_F_detail = DetailFuseLayer(feature_VIS_detail, feature_I_detail)

            data_Fuse, _ = Decoder(data_VIS, feature_F_ll, feature_F_detail)
            data_Fuse = (data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            img_save(fi.astype(np.uint8), img_name.split(sep='.')[0], test_out_folder)
        
    
    #for rgb fusion results
        # for img_name in os.listdir(os.path.join(test_folder, 'ir')):
        #     data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
        #                   np.newaxis, np.newaxis, ...] / 255.0
        #     data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='YCrCb'))[0][
        #                    np.newaxis, np.newaxis, ...] / 255.0
        #
        #     data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
        #     _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))
        #     data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
        #     data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        #
        #     feature_V_B, feature_V_D = Encoder(data_VIS)
        #     feature_I_B, feature_I_D = Encoder(data_IR)
        #     feature_F_B = BaseFuseLayer(feature_V_B , feature_I_B)
        #     feature_F_D = DetailFuseLayer(feature_V_D , feature_I_D)
        #     data_Fuse, _ = Decoder(data_VIS, feature_V_B, feature_F_D)
        #     data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
        #     fi = np.squeeze((data_Fuse * 255).cpu().numpy())
        #     fi = fi.astype(np.uint8)
        #     ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
        #     rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
        #     img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)

    eval_folder=test_out_folder
    ori_img_folder=test_folder

    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            if vi.shape[0] == fi.shape[0] and vi.shape[1] == fi.shape[1]:
                vi = vi
                ir = ir
            else:
                vi = vi[0:fi.shape[0], 0:fi.shape[1]]
                ir = ir[0:fi.shape[0], 0:fi.shape[1]]
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))
            )
    print("="*80)
