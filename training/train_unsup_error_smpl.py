from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys

sys.path.append('./auxiliary/')
from dataset import *
from model_err import *
import my_utils

my_utils.plant_seeds(randomized_seed=False)
from sklearn.neighbors import NearestNeighbors
from ply import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import json
import datetime
import visdom
from LaplacianLoss import *
from knn_cuda import KNN
from smpl_torch_batch import SMPLModel
# =============PARAMETERS======================================== #
torch.set_default_tensor_type('torch.FloatTensor')
lambda_laplace = 0.005
lambda_ratio = 0.005

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=48)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
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
vis = visdom.Visdom(port=8888, env=opt.env)
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



smpl = SMPLModel(device=torch.device('cuda'),model_path='./data/basicModel_m_lbs_10_207_0_v1.1.0.pkl')


# takes cuda torch variable repeated batch time

vertices =smpl.v_template#network.mesh.vertices #smpl.v_template#network.mesh.vertices
vertices=vertices.cpu().detach().numpy()
network = AE_AtlasNet_Humans(inputmesh=vertices)
faces = network.mesh.faces#smpl.faces
faces = [faces for i in range(opt.batchSize)]
faces = np.array(faces)
#faces.dtype="float32"
faces = torch.from_numpy(faces).cuda()
print("face",faces)
print("vertices",vertices)
#vertices = np.array(vertices)
vertices = [vertices for i in range(opt.batchSize)]
vertices = np.array(vertices)
#vertices = np.array(vertices)
#vertices=vertices.numpy()
#vertices.dtype="float32"
print(type(vertices))
vertices = torch.from_numpy(vertices).cuda()
toref = opt.laplace  # regularize towards 0 or template

# Initialize Laplacian Loss
laplaceloss = LaplacianLoss(faces, vertices, toref)

tttt=laplaceloss(vertices)
print("laplaceloss",tttt)
network.cuda()  # put network on GPU
network.apply(my_utils.weights_init)  # initialization of the weight
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)

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
neigh = NearestNeighbors(n_neighbors=1, algorithm='auto')
# Load all the points from the template
template_points = smpl.v_template.clone()#network.mesh.vertices #smpl.v_template#network.mesh.vertices
#template_points=template_points.cpu().detach().numpy()
template_points = template_points.unsqueeze(0).expand(opt.batchSize, template_points.size(0), template_points.size(
    1))  # have to have two stacked template because of weird error related to batchnorm
template_points = Variable(template_points, requires_grad=False)
template_points = template_points.cuda()


def Tran_loss(Translate_matrix, template_points_toTra, Target_points):# tran: b n p tem b p n  Tar  b n p

    template_points_toTra=template_points_toTra.transpose(2,1)
    Recon_points = Tran_points(Translate_matrix, template_points_toTra, None)
    loss = torch.mean((Recon_points - Target_points) ** 2)
    return loss


def Tran_points_rig(Translate_matrix, Rig_list):
    #  start=time.time()
    pointtran = Translate_matrix
    indices_r = torch.tensor([0, 1, 2]).cuda()
    Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
    indices_t = torch.tensor([3, 4, 5]).cuda()
    Point_T = torch.index_select(pointtran, dim=2, index=indices_t)
    Point_R_list=torch.rand((Translate_matrix.size(0),1,3)).cuda()
    for idx in Rig_list:
        Point_R_i = torch.mean(Point_R[:, idx, :], dim=1, keepdim=True)
        Point_R_list=torch.cat((Point_R_list,Point_R_i),1)
    Point_R_list=Point_R_list[:,1:,:]
    Point_T_i = torch.mean(Point_T[:, :, :], dim=1, keepdim=True)
    Point_T_i=Point_T_i.view(Point_T_i.size(0),-1).cuda()
    #model = SMPLModel(device=torch.device('cuda'))
    betas=torch.zeros((Translate_matrix.size(0),300),dtype=torch.float).cuda()
    #print("betas",betas.shape)
    #print("Point_R_list",Point_R_list.shape)
    #print("Point_T_i",Point_T_i.shape)
    pointsReconstructed,jnts=smpl(betas,Point_R_list,Point_T_i)
    return pointsReconstructed


def Tran_points(Translate_matrix, template_points_toTra, train_sig):
    if train_sig:
        pointsReconstructed = template_points_toTra + Translate_matrix
    else:
        pointsReconstructed = template_points_toTra - Translate_matrix
    return pointsReconstructed


Rig_list = []
for i in range(24):
    path="./data/output/jnts"+str(i)+".npy"
    jnt_t = np.load(path)
    Rig_list.append(jnt_t)

'''
head = np.load("./data/output/head_indices.npy")
Rig_list.append(head)
left_arm_down = np.load("./data/output/left_arm_down_indices.npy")
Rig_list.append(left_arm_down)
left_arm = np.load("./data/output/left_arm_indices.npy")
Rig_list.append(left_arm)
left_foot = np.load("./data/output/left_foot_indices.npy")
Rig_list.append(left_foot)
left_hand = np.load("./data/output/left_hand_indices.npy")
Rig_list.append(left_hand)
left_leg_down = np.load("./data/output/left_leg_down_indices.npy")
Rig_list.append(left_leg_down)
left_leg = np.load("./data/output/left_leg_indices.npy")
Rig_list.append(left_leg)
right_arm_down = np.load("./data/output/right_arm_down_indices.npy")
Rig_list.append(right_arm_down)
right_arm = np.load("./data/output/right_arm_indices.npy")
Rig_list.append(right_arm)
right_foot = np.load("./data/output/right_foot_indices.npy")
Rig_list.append(right_foot)
right_hand = np.load("./data/output/right_hand_indices.npy")
Rig_list.append(right_hand)
right_leg_down = np.load("./data/output/right_leg_down_indices.npy")
Rig_list.append(right_leg_down)
right_leg = np.load("./data/output/right_leg_indices.npy")
Rig_list.append(right_leg)
torso = np.load("./data/output/torso_indices.npy")
Rig_list.append(torso)'''
# =============start of the learning loop ======================================== #
for epoch in range(opt.nepoch):
    if epoch==40:
       lambda_laplace = 0.005
       lambda_ratio = 0.005
    if epoch==60:
       lambda_laplace = 0.005
       lambda_ratio = 0.005

    if epoch == 80:
        lambda_laplace = 0.005
        lambda_ratio = 0.005
        lrate = lrate / 10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network.parameters(), lr=lrate)
    if epoch == 90:
        lrate = lrate / 10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network.parameters(), lr=lrate)

    # TRAIN MODE
    train_loss_L2_smpl.reset()
    network.train()
    if epoch < 1:
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
                points = points.cuda()
                points_tra = network.forward_rig(points)
                pointsReconstructed = Tran_points_rig(points_tra,
                                                      Rig_list=Rig_list)  # forward pass
                # pointsReconstructed =network(points)
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
            # pointsReconstructed = network(points)
            pointsReconstructed = Tran_points_rig(points_tra,
                                                  Rig_list=Rig_list)  # forward pass
            # compute the laplacian loss

            regul = laplaceloss(pointsReconstructed)
            dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
            loss_net = (torch.mean(dist1)) + (
                torch.mean(dist2)) #+ lambda_laplace * regul + lambda_ratio * compute_score(pointsReconstructed,
                                                                                          # network.mesh.faces,
                                                                                          # target)
            #print("regul",regul)
            #print("dist1",dist1)
            #print("dist2",dist2)
            #print("score",compute_score(pointsReconstructed,network.mesh.faces,target))
            loss_net.backward()
            train_loss_L2_smpl.update(loss_net.item())
            optimizer.step()  # gradient update
            print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))
    else:
        if epoch == 1:
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
                points = points.cuda()
                points_tra_RIG = network.forward_rig(points)
                pointsReconstructed_RIG = Tran_points_rig(points_tra_RIG,
                                                          Rig_list=Rig_list)  # 刚性变换
                points_tra = network.forward(points)

                pointsReconstructed = Tran_points(points_tra, pointsReconstructed_RIG, train_sig=1)  # 非刚性变换
                # pointsReconstructed =network(points)
                loss_net = torch.mean((pointsReconstructed - template_points) ** 2)
                loss_net.backward()
                optimizer.step()  # gradient update
                print('init [%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32,  loss_net.item()))
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            points, fn, idx, _ = data
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()

            points_tra_RIG = network.forward_rig(points)
            pointsReconstructed_RIG = Tran_points_rig(points_tra_RIG,Rig_list=Rig_list)  # 刚性变换
            regul_rig = laplaceloss(pointsReconstructed_RIG)
            dist1_rig, dist2_rig = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed_RIG)
            print("dist_rig loss",torch.mean(dist1_rig))
            print("dist_rig loss2",torch.mean(dist2_rig))
            loss_net_rig = (torch.mean(dist1_rig)) + (
                torch.mean(dist2_rig)) #+ lambda_laplace * regul_rig + lambda_ratio * compute_score(pointsReconstructed_RIG,
                                                                                         #  network.mesh.faces,
                                                                                         #  target)
            print("rig loss",loss_net_rig)
            #points_tra = network.forward(points)

            #pointsReconstructed = Tran_points(points_tra, pointsReconstructed_RIG, train_sig=1)
            # compute the laplacian loss

            #regul = laplaceloss(pointsReconstructed)
            #print("regul loss",regul)
            #dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)

            if epoch > 75:
                knn = KNN(k=1, transpose_mode=True)
                dist, indx = knn(points.transpose(2, 1), pointsReconstructed)
                # print(indx)
                indx_t = indx.view(opt.batchSize, 6890).unsqueeze(-1).expand(opt.batchSize, 6890, 3).cuda()
                points = points.transpose(2, 1).gather(1, indx_t)

                # import IPython;IPython.embed()
                points = points.transpose(2, 1)
                recon_loss = Tran_loss(points_tra, points, pointsReconstructed_RIG)
                loss_net = (torch.mean(dist1)) + (
                    torch.mean(dist2)) + lambda_laplace * regul + lambda_ratio * compute_score(pointsReconstructed,
                                                                                               network.mesh.faces,
                                                                                               target) +  recon_loss+loss_net_rig.item()
            else:
                #print("dist loss",torch.mean(dist1))
                #print("dist loss2",torch.mean(dist2))
                print("score loss",compute_score(pointsReconstructed_RIG,network.mesh.faces,target))
                #loss_net = (torch.mean(dist1)) + (
                   # torch.mean(dist2)) + lambda_laplace * regul + lambda_ratio * compute_score(pointsReconstructed,
                                                                                              # network.mesh.faces,
                                                                                             #  target)+loss_net_rig.item()
            loss_net_rig.backward()
            train_loss_L2_smpl.update(loss_net_rig.item())
            optimizer.step()  # gradient update
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
            pointsReconstructed_RIG = Tran_points_rig(points_tra_RIG,Rig_list=Rig_list)  # 刚性变换
            points_tra = network.forward(points)

            pointsReconstructed = Tran_points(points_tra, pointsReconstructed_RIG, train_sig=1)
            # points_tra = network.forward(points)
            # pointsReconstructed = network(points)

            # pointsReconstructed = Tran_points(points_tra, template_points_temp,train_sig=1)  # forward pass
            loss_net = torch.mean(
                (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
            val_loss_L2_smpl.update(loss_net.item())
            # VIZUALIZE
            """
            if i % 10 == 0:
                vis.scatter(X=points.transpose(2, 1).contiguous()[0].data.cpu(),
                            win='Test_smlp_input',
                            opts=dict(
                                title="Test_smlp_input",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsReconstructed[0].data.cpu(),
                            win='Test_smlp_output',
                            opts=dict(
                                title="Test_smlp_output",
                                markersize=2,
                            ),
                            )
            """
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
        torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))

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
