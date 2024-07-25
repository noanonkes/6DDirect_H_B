import time
import os
import argparse
import cv2
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.backends import cudnn
from torchvision import transforms

from model import SixDRepNet
import utils
import datasets


from scipy.spatial.transform import Rotation

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--data_dir',
                        dest='data_dir', help='Directory path for data.',
                        default='datasets/AFLW2000', type=str)
    parser.add_argument('--filename_list',
                        dest='filename_list',
                        help='Path to text file containing relative paths for every example.',
                        default='datasets/AFLW2000/files.txt', type=str)  # datasets/BIWI_noTrack.npz
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--batch_size',
                        dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--show_viz',
                        dest='show_viz', help='Save images with pose cube.',
                        default=False, type=bool)
    parser.add_argument('--dataset',
                        dest='dataset', help='Dataset type.',
                        default='AFLW2000', type=str)


    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

class GeodesicLoss(nn.Module):
    """ 
    Input: Rotations matrices of shape batch x 3 x 3
           Both matrices are orthogonal rotation matrices

    Output: Average theta between 0 to pi radians / pi.
    """
    def __init__(self, eps=1e-7, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    def forward(self, m1, m2):
        if len(m1.shape) < 3:
            m1 = m1[None]
        if len(m2.shape) < 3:
            m2 = m2[None]

        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
        
        if self.reduction == "mean":
            return torch.mean(theta) / np.pi
        return theta / np.pi
    

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='',
                        deploy=True,
                        pretrained=False,
                        gpu_id=gpu)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(
                                              224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations, train_mode = False)
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        num_workers=2)


    # Load snapshot
    saved_state_dict = torch.load(snapshot_path, map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)    
    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    
    total, total_front = 0, 0
    yaw_error = pitch_error = roll_error = .0
    yaw_error_front = pitch_error_front = roll_error_front = .0
    v1_err = v2_err = v3_err = .0

    geo_loss = 0.
    geo_func = GeodesicLoss()

    angular_error = 0.

    with torch.no_grad():
        
        time_list = []
        for i, (images, r_label, cont_labels, name) in enumerate(tqdm(test_loader)):
            images = torch.Tensor(images).cuda(gpu)
            total += cont_labels.size(0)
                        
            # gt matrix
            R_gt = r_label

            # gt euler
            y_gt_deg = cont_labels[:, 0]*180/np.pi
            p_gt_deg = cont_labels[:, 1]*180/np.pi
            r_gt_deg = cont_labels[:, 2]*180/np.pi
            
            start_time = time.time()
            
            R_pred = model(images)
            geo_loss += geo_func(R_pred, R_gt.cuda(gpu))

            R_pred_vec = R_pred[0] @ torch.tensor([0., 0., -1.])
            R_gt_vec = R_gt[0] @ torch.tensor([0., 0., -1.])

            R_pred_vec = R_pred_vec / torch.linalg.norm(R_pred_vec)
            R_gt_vec = R_gt_vec / torch.linalg.norm(R_gt_vec)

            angular_error += torch.acos(torch.clip(torch.sum(R_pred_vec * R_gt_vec), 0., 1.))

            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred, full_range=True)*180/np.pi
            p_pred_deg = euler[:, 0].cpu()  # range (-90, 90)
            y_pred_deg = euler[:, 1].cpu()  # range (-180, 180)
            r_pred_deg = euler[:, 2].cpu()  # range (-90, 90)

            end_time = time.time()
            time_list.append(end_time - start_time)

            # R_pred = R_pred.cpu()
            # v1_err += torch.sum(torch.acos(torch.clamp(
                # torch.sum(R_gt[:, 0] * R_pred[:, 0], 1), -1, 1)) * 180/np.pi)
            # v2_err += torch.sum(torch.acos(torch.clamp(
                # torch.sum(R_gt[:, 1] * R_pred[:, 1], 1), -1, 1)) * 180/np.pi)
            # v3_err += torch.sum(torch.acos(torch.clamp(
                # torch.sum(R_gt[:, 2] * R_pred[:, 2], 1), -1, 1)) * 180/np.pi)
    
            '''
            for full-range angles
            error calculation v2: wrapped loss by us, we cannot use the 180 term for the yaw
            '''
            pitch_error_temp = torch.sum(torch.min(torch.stack((
                torch.abs(p_gt_deg - p_pred_deg), 
                torch.abs(p_pred_deg + 360 - p_gt_deg), 
                torch.abs(p_pred_deg - 360 - p_gt_deg), 
                torch.abs(p_pred_deg + 180 - p_gt_deg), 
                torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
            yaw_error_temp = torch.sum(torch.min(torch.stack((
                torch.abs(y_gt_deg - y_pred_deg), 
                torch.abs(y_pred_deg + 360 - y_gt_deg), 
                torch.abs(y_pred_deg - 360 - y_gt_deg))), 0)[0])
            roll_error_temp = torch.sum(torch.min(torch.stack((
                torch.abs(r_gt_deg - r_pred_deg), 
                torch.abs(r_pred_deg + 360 - r_gt_deg), 
                torch.abs(r_pred_deg - 360 - r_gt_deg),
                torch.abs(r_pred_deg + 180 - r_gt_deg), 
                torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])
            
            
            pitch_error_temp = min(pitch_error_temp, abs(360 - pitch_error_temp))
            yaw_error_temp = min(yaw_error_temp, abs(360 - yaw_error_temp))
            roll_error_temp = min(roll_error_temp, abs(360 - roll_error_temp))
                
            pitch_error += pitch_error_temp
            yaw_error += yaw_error_temp
            roll_error += roll_error_temp
            if len(y_gt_deg) == 1 and abs(y_gt_deg[0]) < 90:
                total_front += 1
                pitch_error_front += pitch_error_temp
                yaw_error_front += yaw_error_temp
                roll_error_front += roll_error_temp
                
            if args.show_viz:
                name = name[0]
                if args.dataset == 'AFLW2000':
                    cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
                   
                elif args.dataset == 'BIWI':
                    vis = np.uint8(name)
                    h,w,c = vis.shape
                    vis2 = cv2.CreateMat(h, w, cv2.CV_32FC3)
                    vis0 = cv2.fromarray(vis)
                    cv2.CvtColor(vis0, vis2, cv2.CV_GRAY2BGR)
                    cv2_img = cv2.imread(vis2)
                utils.draw_axis(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], tdx=200, tdy=200, size=100)
                #utils.plot_pose_cube(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=200)
                cv2.imshow("Test", cv2_img)
                cv2.waitKey(5)
                cv2.imwrite(os.path.join('output/img/',name+'.png'),cv2_img)
        
        print("Inference time per image: ", sum(time_list) / len(time_list))
                    
        print('[Total heads: %d] Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f, Geo: %.4f' % (total,
            yaw_error / total, pitch_error / total, roll_error / total,
            (yaw_error + pitch_error + roll_error) / (total * 3), geo_loss / total))
        
        # print('Vec1: %.4f, Vec2: %.4f, Vec3: %.4f, VMAE: %.4f' % (
        #     v1_err / total, v2_err / total, v3_err / total,
        #     (v1_err + v2_err + v3_err) / (total * 3)))

        if total_front != 0:
            print('[Front faces: %d] Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (total_front,
                yaw_error_front / total_front, pitch_error_front / total_front, roll_error_front / total_front,
                (yaw_error_front + pitch_error_front + roll_error_front) / (total_front * 3)))
