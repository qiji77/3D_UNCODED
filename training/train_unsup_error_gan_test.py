from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
import multiprocessing
sys.path.append('./auxiliary/')
from dataset import *

import my_utils

my_utils.plant_seeds(randomized_seed=False)
from sklearn.neighbors import NearestNeighbors
from ply import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from model_err_nfeaPointnet2_test import *
import json
import datetime
from LaplacianLoss import *
from knn_cuda import KNN
knn = KNN(k=1, transpose_mode=True)
import time
import warnings
warnings.filterwarnings("ignore")
# =============PARAMETERS======================================== #
lambda_laplace = 0.005
lambda_ratio = 0.005

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--nepoch', type=int, default=75, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--env', type=str, default="3DCODED_unsupervised", help='visdom environment')
parser.add_argument('--laplace', type=int, default=1, help='regularize towords 0 curvature, or template curvature')

opt = parser.parse_args()
print(opt)
# ========================================================== #

# =============DEFINE CHAMFER LOSS======================================== #
sys.path.append("./extension/")
import dist_chamfer as ext

distChamfer = ext.chamferDist()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
L2curve_train_smpl = []
L2curve_val_smpl = []

# meters to record stats on learning
train_loss_L2_smpl = my_utils.AverageValueMeter()
val_loss_L2_smpl = my_utils.AverageValueMeter()
tmp_val_loss = my_utils.AverageValueMeter()
# ========================================================== #


# ===================CREATE DATASET================================= #
dataset = SURREAL(train=True, regular_sampling=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)
dataset_smpl_test = SURREAL(train=False)
dataloader_smpl_test = torch.utils.data.DataLoader(dataset_smpl_test, batch_size=opt.batchSize,
                                                   shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #


network = AE_AtlasNet_Humans()
GenNetwork=GenPointsFeatures()
faces = network.mesh.faces
faces = [faces for i in range(opt.batchSize)]
faces = np.array(faces)
faces = torch.from_numpy(faces).cuda()
# takes cuda torch variable repeated batch time

vertices = network.mesh.vertices
vertices = [vertices for i in range(opt.batchSize)]
vertices = np.array(vertices)
vertices = torch.from_numpy(vertices).cuda()
toref = opt.laplace  # regularize towards 0 or template

# Initialize Laplacian Loss
laplaceloss = LaplacianLoss(faces, vertices, toref)

laplaceloss(vertices)
network.cuda()  # put network on GPU
network.apply(my_utils.weights_init)  # initialization of the weight
GenNetwork.cuda()#特征生成网络和判别器
GenNetwork.apply(my_utils.weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)
optimizer_gen=optim.Adam(GenNetwork.parameters(),lr=lrate)
with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')


# ========================================================== #

def init_regul(source):
    sommet_A_source = source.vertices[source.faces[:, 0]]
    sommet_B_source = source.vertices[source.faces[:, 1]]
    sommet_C_source = source.vertices[source.faces[:, 2]]
    target = []
    target.append(np.sqrt(np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt(np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt(np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    return target


target = init_regul(network.mesh)
target = np.array(target)
target = torch.from_numpy(target).float().cuda()
target = target.unsqueeze(1).expand(3, opt.batchSize, -1)


def compute_score(points, faces, target):
    score = 0
    sommet_A = points[:, faces[:, 0]]
    sommet_B = points[:, faces[:, 1]]
    sommet_C = points[:, faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] - 1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] - 1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] - 1)
    return torch.mean(score)


# ========================================================== #
# Load all the points from the template
template_points = network.vertex.clone() #6890 3  模板模型
template_points = template_points.unsqueeze(0).expand(opt.batchSize, template_points.size(0), template_points.size(
    1))  # have to have two stacked template because of weird error related to batchnorm
template_points = Variable(template_points, requires_grad=False)
template_points = template_points.cuda()

def Tran_loss(Translate_matrix, template_points_toTra, Target_points):# tran: b n p tem b p n  Tar  b n p
#输入：变换位姿，模板点，目标点
    template_points_toTra=template_points_toTra.transpose(2,1)
    Recon_points = Tran_points(Translate_matrix, template_points_toTra, None)
    loss = torch.mean((Recon_points - Target_points) ** 2)
    return loss


def Tran_points_rig(Translate_matrix, template_points_toTra, train_sig, Rig_list):
    #  start=time.time()
    pointtran = Translate_matrix
    indices_r = torch.tensor([0, 1, 2]).cuda()
    Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
    indices_theat = torch.tensor([3]).cuda()
    Point_theat = torch.index_select(pointtran, dim=2, index=indices_theat)
    indices_t = torch.tensor([4, 5, 6]).cuda()
    Point_T = torch.index_select(pointtran, dim=2, index=indices_t)
    if train_sig:
        # print("train is ", train_sig)
        rand_grid = template_points_toTra
        rand_grid = rand_grid.view(points.size(0), -1, 3).transpose(2,
                                                                    1).contiguous()  # batch , 2500, 3 -> batch, 6980, 3
    else:
        # print("template_points_toTra size is ",template_points_toTra.size())
        rand_grid = template_points_toTra.transpose(2, 1) - Point_T
        rand_grid = rand_grid.view(points.size(0), -1, 3).transpose(2,
                                                                    1).contiguous()  # batch , 2500, 3 -> batch, 6980, 3
        Point_theat = -1 * Point_theat

    #Point_R_clone = Point_R.clone()
    #Point_theat_clone = Point_theat.clone()
    #Point_T_clone = Point_T.clone()
    #items = [for idx in Rig_list]
    #p = multiprocessing.Pool(24)
    #start = timeit.default_timer()
    #b = p.map(CalValue, items)
    #p.close()
    #p.join()
    #for idx in Rig_list:
        # idx=torch.from_numpy(idx)
        # Point_R_clone_t=Point_R_clone.cpu().detach().numpy()
    Point_R[:, Rig_list, :] = torch.mean(Point_R[:, Rig_list, :], dim=1, keepdim=True)
    Point_theat[:, Rig_list, :] = torch.mean(Point_theat[:, Rig_list, :], dim=1, keepdim=True)
    Point_T[:, Rig_list, :] = torch.mean(Point_T[:, Rig_list, :], dim=1, keepdim=True)
    Rota_point = rand_grid  # 要旋转的点  也就是 基础模板点
    Rota_point_t = Rota_point.transpose(2, 1).cuda()
    # Rota_point_t2=torch.zeros(Rota_point_t.size(0),Rota_point_t.size(1),Rota_point_t.size(2)).cuda()
    Cos_theat = torch.cos(Point_theat)
    point_r_list = Point_R.view(Point_R.size(0), -1, 3, 1)
    K_list = point_r_list / torch.sqrt(torch.matmul(torch.transpose(point_r_list, 3, 2), point_r_list))
    K_list_view = K_list.view(K_list.size(0), -1, 3, 1)
    Rota_point_t_view = Rota_point_t.view(Rota_point_t.size(0), -1, 3, 1)
    # print(K_list.size())
    Sin_theat = torch.sin(Point_theat)

    K_list = K_list.view(K_list.size(0), -1, 3)
    K_list_X1 = K_list[:, :, [1, 2, 0]]
    K_list_X2 = K_list[:, :, [2, 0, 1]]
    V_list_X1 = Rota_point_t[:, :, [2, 0, 1]]
    V_list_X2 = Rota_point_t[:, :, [1, 2, 0]]
    k_p_v = torch.matmul(torch.transpose(K_list_view, 3, 2), Rota_point_t_view)
    first_term = Cos_theat * Rota_point_t
    K_list = K_list.view(K_list.size(0), -1, 3)
    second_term = (1 - Cos_theat) * (k_p_v.view(k_p_v.size(0), -1, 1) * K_list)
    third_term = Sin_theat * (K_list_X1 * V_list_X1 - K_list_X2 * V_list_X2)
    Rota_point_t2 = first_term + second_term + third_term
    if train_sig:
        pointsReconstructed = Rota_point_t2 + Point_T
    else:
        pointsReconstructed = Rota_point_t2
    return pointsReconstructed


def Tran_points(Translate_matrix, template_points_toTra, train_sig):
    if train_sig:
        pointsReconstructed = template_points_toTra + Translate_matrix
    else:
        pointsReconstructed = template_points_toTra - Translate_matrix
    return pointsReconstructed
    
def Tran_points_noise(template_cloud,Point_R,Point_theat,Point_T):#刚性变换  Translate_matrix:B N R&T  template_cloud: B N P
    #template_cloud=template_cloud.transpose(2,1).contiguous()
    template_cloud=template_cloud.contiguous()
    template_cloud =template_cloud.clone().detach().requires_grad_(True)# torch.tensor(template_cloud, dtype=torch.float32)
    Point_R = torch.tensor(Point_R, dtype=torch.float32)
    
    Point_theat = torch.tensor(Point_theat, dtype=torch.float32)
    Point_T= torch.tensor(Point_T, dtype=torch.float32)
    Point_R=Point_R.unsqueeze(1).expand(template_cloud.size(0),template_cloud.size(1), 3).cuda()
    Point_theat = Point_theat.unsqueeze(1).expand(template_cloud.size(0), template_cloud.size(1), 1).cuda()
    Point_T = Point_T.unsqueeze(1).expand(template_cloud.size(0), template_cloud.size(1), 3).cuda()
    Rota_point = template_cloud  # 要旋转的点  也就是 基础模板点
    Rota_point_t = Rota_point.cuda()
    Cos_theat = torch.cos(Point_theat)
    point_r_list = Point_R.view(Point_R.size(0), -1, 3, 1)
    K_list = point_r_list / torch.sqrt(torch.matmul(torch.transpose(point_r_list, 3, 2), point_r_list))
    K_list_view = K_list.view(K_list.size(0), -1, 3, 1)
    Rota_point_t_view = Rota_point_t.view(Rota_point_t.size(0), -1, 3, 1)
    Sin_theat = torch.sin(Point_theat)
    K_list = K_list.view(K_list.size(0), -1, 3)
    K_list_X1 = K_list[:, :, [1, 2, 0]]
    K_list_X2 = K_list[:, :, [2, 0, 1]]
    V_list_X1 = Rota_point_t[:, :, [2, 0, 1]]
    V_list_X2 = Rota_point_t[:, :, [1, 2, 0]]
    k_p_v = torch.matmul(torch.transpose(K_list_view, 3, 2), Rota_point_t_view)
    first_term = Cos_theat * Rota_point_t
    K_list = K_list.view(K_list.size(0), -1, 3)
    second_term = (1 - Cos_theat) * (k_p_v.view(k_p_v.size(0), -1, 1) * K_list)
    third_term = Sin_theat * (K_list_X1 * V_list_X1 - K_list_X2 * V_list_X2)
    Rota_point_t2 = first_term + second_term + third_term
    pointsReconstructed = Rota_point_t2 + Point_T
    return pointsReconstructed
def MakeNoise(batch):# 设置随机扰动
    Point_R = np.random.uniform(-1, 1, size=(batch,3))
    Point_Theta = np.random.uniform(-0.05, 0.05, size=(batch,1))
    Point_xyz=np.random.uniform(-0.0005, 0.0005, size=(batch,3))
    Point_R=torch.from_numpy(Point_R).cuda()
    Point_Theta = torch.from_numpy(Point_Theta).cuda()
    Point_xyz = torch.from_numpy(Point_xyz).cuda()
    return Point_R,Point_Theta,Point_xyz
#获得part指数
Rig_list = []
head = np.load("./data/output/head_indices.npy")
Rig_list.extend(head)
left_arm_down = np.load("./data/output/left_arm_down_indices.npy")
Rig_list.extend(left_arm_down)
left_arm = np.load("./data/output/left_arm_indices.npy")
Rig_list.extend(left_arm)
left_foot = np.load("./data/output/left_foot_indices.npy")
Rig_list.extend(left_foot)
left_hand = np.load("./data/output/left_hand_indices.npy")
Rig_list.extend(left_hand)
left_leg_down = np.load("./data/output/left_leg_down_indices.npy")
Rig_list.extend(left_leg_down)
left_leg = np.load("./data/output/left_leg_indices.npy")
Rig_list.extend(left_leg)
right_arm_down = np.load("./data/output/right_arm_down_indices.npy")
Rig_list.extend(right_arm_down)
right_arm = np.load("./data/output/right_arm_indices.npy")
Rig_list.extend(right_arm)
right_foot = np.load("./data/output/right_foot_indices.npy")
Rig_list.extend(right_foot)
right_hand = np.load("./data/output/right_hand_indices.npy")
Rig_list.extend(right_hand)
right_leg_down = np.load("./data/output/right_leg_down_indices.npy")
Rig_list.extend(right_leg_down)
right_leg = np.load("./data/output/right_leg_indices.npy")
Rig_list.extend(right_leg)
torso = np.load("./data/output/torso_indices.npy")
Rig_list.extend(torso)
# =============start of the learning loop ======================================== #
for epoch in range(opt.nepoch):
    if epoch == 80:
        lambda_laplace = 0.005
        lambda_ratio = 0.005
        lrate = lrate / 10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network.parameters(), lr=lrate)
        optimizer_gen=optim.Adam(GenNetwork.parameters(),lr=lrate)
    if epoch == 90:
        lrate = lrate / 10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network.parameters(), lr=lrate)
        optimizer_gen = optim.Adam(GenNetwork.parameters(), lr=lrate)

    # TRAIN MODE
    train_loss_L2_smpl.reset()
    GenNetwork.train()
    network.train()
    if epoch <0:
        if epoch == 0:
            # initialize reconstruction to be same as template to avoid symmetry issues
            init_step = 0
            for i, data in enumerate(dataloader, 0):
                if (init_step > 1000):
                    break
                init_step = init_step + 1
                optimizer.zero_grad()
                points, fn, idx, _ = data
                points = points.transpose(2, 1).contiguous()
                print("point_size is", points.size())
                points = points.cuda() #B 3 N
                points_tra = network.forward_rig(points)#刚性变换网络
                pointsReconstructed = Tran_points_rig(points_tra, template_points, train_sig=1,
                                                      Rig_list=Rig_list)  # 刚性变换
                loss_net = torch.mean((pointsReconstructed - template_points) ** 2)
                loss_net.backward()
                optimizer.step()  # gradient update
                print('init [%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            points, fn, idx, _ = data
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            points_tra = network.forward_rig(points)
            pointsReconstructed = Tran_points_rig(points_tra, template_points, train_sig=1,
                                                  Rig_list=Rig_list)  # forward pass
            # compute the laplacian loss

            regul = laplaceloss(pointsReconstructed)
            dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
            loss_net = (torch.mean(dist1)) + (
                torch.mean(dist2)) + lambda_laplace * regul + lambda_ratio * compute_score(pointsReconstructed,
                                                                                           network.mesh.faces,
                                                                                           target)
            loss_net.backward()
            train_loss_L2_smpl.update(loss_net.item())
            optimizer.step()  # gradient update
            print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))
    else:
        if epoch == -1:
            # initialize reconstruction to be same as template to avoid symmetry issues
            init_step = 0
            for i, data in enumerate(dataloader, 0):
                if (init_step > 1000):
                    break
                init_step = init_step + 1

                optimizer.zero_grad()
                points, fn, idx, _ = data
                points = points.transpose(2, 1).contiguous()
               
                points = points.cuda()
                points_tra_RIG = network.forward_rig(points)
                pointsReconstructed_RIG = Tran_points_rig(points_tra_RIG, template_points, train_sig=1,
                                                          Rig_list=Rig_list)  # 刚性变换
                points_tra = network.forward(points)

                pointsReconstructed = Tran_points(points_tra, pointsReconstructed_RIG, train_sig=1)  # 非刚性变换
                # pointsReconstructed =network(points)
                loss_net = torch.mean((pointsReconstructed - template_points) ** 2)
                loss_net.backward()
                optimizer.step()  # gradient update
                print('init [%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32,  loss_net.item()))
        for i, data in enumerate(dataloader, 0):
            
            beforetime=time.time()
            points, fn, idx, _ = data
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            #points_clone=points.clone().detach()
            
            
            rand_id=np.random.randint(0,6890,size=30)
            rand_id=torch.from_numpy(rand_id).cuda()
            #rand_id=rand_id.unsqueeze(-1).expand(opt.batchSize, rand_id.size(1), 3).cuda()

            #### Appendix ####
            #### var.detach() = Variable(var.data), difference in computing trigger
            points_tra_RIG = network.forward_rig(points)
            pointsReconstructed_RIG = Tran_points_rig(points_tra_RIG, template_points, train_sig=1,
                                                      Rig_list=Rig_list)  # 刚性变换
           
            regul_rig = laplaceloss(pointsReconstructed_RIG)
            dist1_rig, dist2_rig = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed_RIG)
            loss_net_rig = (torch.mean(dist1_rig)) + (
                torch.mean(dist2_rig)) + lambda_laplace * regul_rig + lambda_ratio * compute_score(pointsReconstructed_RIG,
                                                                                           network.mesh.faces,
                                                                                           target)
            points_tra = network.forward(points)

            pointsReconstructed = Tran_points(points_tra, pointsReconstructed_RIG, train_sig=1)
            regul = laplaceloss(pointsReconstructed)
            dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
            #找最近点
            dist, indx = knn(points.transpose(2, 1),pointsReconstructed)
            indx_t = indx.view(opt.batchSize, 6890).unsqueeze(-1).expand(opt.batchSize, 6890, 3).cuda()
            #pointsReconstructed = pointsReconstructed.gather(1, indx_t)
            points = points.transpose(2, 1).gather(1, indx_t)
            noise_r,noise_theta,noise_xyz=MakeNoise(opt.batchSize)
            Points_noise=Tran_points_noise(points,noise_r,noise_theta,noise_xyz)
            #Points_noise=Points_noise.transpose(2, 1)
            #真实值
            #x=torch.index_select(x, dim=2, index=idx)
            Points_map=GenNetwork.foward_encoder(points)
            Points_noise_map=GenNetwork.foward_encoder(Points_noise)
            Points_fake_map=GenNetwork.foward_encoder(pointsReconstructed.detach())
            Points_map=torch.index_select(Points_map, dim=2, index=rand_id)
            Points_noise_map=torch.index_select(Points_noise_map, dim=2, index=rand_id)
            Points_fake_map=torch.index_select(Points_fake_map, dim=2, index=rand_id)
            label_R=GenNetwork.foward_decoder(Points_map,Points_noise_map)
            label_R=label_R.transpose(2, 1)
            errD_real = 0.5 * torch.mean((label_R - 1) ** 2)  # criterion(output, label)
            #errD_real.backward()
            #扰动值
            label_F = GenNetwork.foward_decoder(Points_map,Points_fake_map)  # Create fake image
            label_F=label_F.transpose(2, 1)
            errD_fake = 0.5 * torch.mean((label_F +1) ** 2)  # criterion(output, label)
            #errD_fake.backward()
            D_loss=errD_real+errD_fake
            optimizer_gen.zero_grad()
            D_loss.backward()
            optimizer_gen.step()
            #optimizer_gen.zero_grad()
            optimizer.zero_grad()
            #errD = errD_fake + errD_real
            #points_tra_RIG = network.forward_rig(points_clone)
            #pointsReconstructed_RIG = Tran_points_rig(points_tra_RIG, template_points, train_sig=1,
            #                                          Rig_list=Rig_list)  # 刚性变换
            
            #points_tra = network.forward(points_clone)
            #pointsReconstructed = Tran_points(points_tra, pointsReconstructed_RIG, train_sig=1)
            
           
            
            #dist, indx = knn(points_clone.transpose(2, 1),pointsReconstructed)
            #indx_t = indx.view(opt.batchSize, 6890).unsqueeze(-1).expand(opt.batchSize, 6890, 3).cuda()
            #pointsReconstructed = pointsReconstructed.gather(1, indx_t)
            #points_clone = points_clone.transpose(2, 1).gather(1, indx_t)
            label_G = GenNetwork.foward_decoder(Points_map,Points_fake_map)  # Create fake images
            label_G=label_G.transpose(2, 1)
            errD_G = 0.05 * torch.mean((label_G) ** 2)  # criterion(output, label)
            if epoch >50:
                
                recon_loss = Tran_loss(points_tra, points_clone.transpose(2, 1), pointsReconstructed_RIG)
                loss_net = (torch.mean(dist1)) + (
                    torch.mean(dist2)) + lambda_laplace * regul + lambda_ratio * compute_score(pointsReconstructed,
                                                                                               network.mesh.faces,
                                                                                               target) +  recon_loss.item()+loss_net_rig.item()+errD_G.item()
            else:
                loss_net = (torch.mean(dist1)) + (
                    torch.mean(dist2)) + lambda_laplace * regul + lambda_ratio * compute_score(pointsReconstructed,
                                                                                               network.mesh.faces,
                                                                                               target)+loss_net_rig.item()+errD_G.item()
            loss_net.backward()
            optimizer.step()
            train_loss_L2_smpl.update(loss_net.item())
            print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))

    with torch.no_grad():
        # val on SMPL data
        network.eval()
        val_loss_L2_smpl.reset()
        for i, data in enumerate(dataloader_smpl_test, 0):
            points, fn, idx, _ = data
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            template_points_temp = template_points[:points.size(0)]
            points_tra_RIG = network.forward_rig(points)
            pointsReconstructed_RIG = Tran_points_rig(points_tra_RIG, template_points_temp, train_sig=1,
                                                      Rig_list=Rig_list)  # 刚性变换
            points_tra = network.forward(points)

            pointsReconstructed = Tran_points(points_tra, pointsReconstructed_RIG, train_sig=1)
            # points_tra = network.forward(points)
            # pointsReconstructed = network(points)

            # pointsReconstructed = Tran_points(points_tra, template_points_temp,train_sig=1)  # forward pass
            loss_net = torch.mean(
                (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
            val_loss_L2_smpl.update(loss_net.item())
            # VIZUALIZE
            print('[%d: %d/%d] test smlp loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))

        L2curve_train_smpl.append(train_loss_L2_smpl.avg)
        L2curve_val_smpl.append(val_loss_L2_smpl.avg)
        """
        vis.line(X=np.column_stack((np.arange(len(L2curve_train_smpl)), np.arange(len(L2curve_val_smpl)))),
                 Y=np.column_stack((np.array(L2curve_train_smpl), np.array(L2curve_val_smpl))),
                 win='loss',
                 opts=dict(title="loss", legend=["L2curve_train_smpl" + opt.env,"L2curve_val_smpl" + opt.env,]))
        vis.line(X=np.column_stack((np.arange(len(L2curve_train_smpl)), np.arange(len(L2curve_val_smpl)))),
                 Y=np.log(np.column_stack((np.array(L2curve_train_smpl), np.array(L2curve_val_smpl)))),
                 win='log',
                 opts=dict(title="log", legend=["L2curve_train_smpl" + opt.env,"L2curve_val_smpl" + opt.env,]))

        """
        # save latest network
        if epoch<75:
            torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))
        else:
            torch.save(network.state_dict(), '%s/network_last75.pth' % (dir_name))
        # dump stats in log file
        log_table = {
            "lambda_laplace": lambda_laplace,
            "lambda_ratio": lambda_ratio,
            "train_loss_L2_smpl": train_loss_L2_smpl.avg,
            "val_loss_L2_smpl": val_loss_L2_smpl.avg,
            "epoch": epoch,
            "lr": lrate,
            "env": opt.env,
        }
        print(log_table)
        with open(logname, 'a') as f:  # open and append
            f.write('json_stats: ' + json.dumps(log_table) + '\n')
