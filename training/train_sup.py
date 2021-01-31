from __future__ import print_function
import sys
sys.path.append('./auxiliary/')
sys.path.append('/app/python/')
sys.path.append('./')
import my_utils
my_utils.plant_seeds(randomized_seed=False)
print("fixed seed")
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import pointcloud_processor
from dataset import *
from model import *
from ply import *
import os
import json
import datetime
import visdom

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--env', type=str, default="3DCODED_supervised", help='visdom environment')
parser.add_argument('--id', type=str, default=None,help='training name')

opt = parser.parse_args()
print(opt)
# ========================================================== #

# =============DEFINE CHAMFER LOSS======================================== #
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer =  ext.chamferDist()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
vis = visdom.Visdom(port=9000, env=opt.env)
now = datetime.datetime.now()
save_path = now.isoformat()
if opt.id is not None:
    dir_name = os.path.join('log',opt.id)
else:
     dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x: '\033[94m' + x + '\033[0m'


L2curve_train_smpl = []
L2curve_val_smlp = []

# meters to record stats on learning
train_loss_L2_smpl = my_utils.AverageValueMeter()
val_loss_L2_smpl = my_utils.AverageValueMeter()
tmp_val_loss = my_utils.AverageValueMeter()
# ========================================================== #


# ===================CREATE DATASET================================= #
dataset = SURREAL(train=True, regular_sampling = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,shuffle=True, num_workers=int(opt.workers))
dataset_smpl_test = SURREAL(train=False)
dataloader_smpl_test = torch.utils.data.DataLoader(dataset_smpl_test, batch_size=5,shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #
network = AE_AtlasNet_Humans()
network.cuda()  # put network on GPU
network.apply(my_utils.weights_init)  # initialization of the weight
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
# ========================================================== #
def Tran_points(points,idx):
  #  start=time.time()
    pointtran = network.forward_idx(points, idx).cuda()
    print("pointtran size is",pointtran.size())
  #  end=time.time()
   # print("network time is ",end-start)
    pointtran_t=pointtran.view(points.size(0), -1, 10)
    #print(pointtran_t.size())
    indices_s = torch.tensor([0,1,2]).cuda()
    Point_S = torch.index_select(pointtran, dim=2, index=indices_s)
    indices_r = torch.tensor([3,4,5]).cuda()
    Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
    indices_theat = torch.tensor([6]).cuda()
    Point_theat = torch.index_select(pointtran, dim=2, index=indices_theat)
    indices_t = torch.tensor([7,8,9]).cuda()
    Point_T = torch.index_select(pointtran, dim=2, index=indices_t)

    if not idx is None:
        idx = idx.view(-1)
        idx = idx.numpy().astype(np.int)

    rand_grid = load_template()  # 6890, 3


    rand_grid=rand_grid.cuda()
    rand_grid = rand_grid[idx, :]  # batch x 2500, 3
    rand_grid = rand_grid.view(points.size(0), -1, 3).transpose(2, 1).contiguous().cuda()  # batch , 2500, 3 -> batch, 6980, 3
    Rota_point = rand_grid.cuda()
    Rota_point_t = Rota_point.transpose(2, 1)
    Rota_point_t2=torch.zeros(Rota_point_t.size(0),Rota_point_t.size(1),Rota_point_t.size(2))
    Cos_theat=torch.cos(Point_theat)
    point_r_list=Point_R.view(Point_R.size(0),-1,3,1)
    K_list=torch.sqrt(torch.matmul(torch.transpose(point_r_list,3,2),point_r_list))*point_r_list
    K_list_view=K_list.view(K_list.size(0),-1,3,1)
    Rota_point_t_view=Rota_point_t.view(Rota_point_t.size(0),-1,3,1)
    #print(K_list.size())
    Sin_theat = torch.sin(Point_theat)
    K_list_X1=torch.cat([K_list[:,:,1], K_list[:,:,2]], dim=2)
    K_list_X1=torch.cat([K_list_X1,K_list[:, :, 0]], dim=2)
    # K_list_X1
    #print("K_list_X1 :",K_list_X1.size())
    K_list_X2 = torch.cat([K_list[:, :, 2], K_list[:, :, 0]],dim=2)
    K_list_X2=torch.cat([K_list_X2, K_list[:, :, 1]],dim=2)
    V_list_X1=Rota_point_t[:,:,[2,0,1]]
    #V_list_X1 = torch.Tensor([Rota_point_t[:, :, 2], Rota_point_t[:, :, 0]],Rota_point_t[:, :, 1])
   # V_list_X1=torch.cat([V_list_X1,Rota_point_t[:, :, 1]],dim=2)
    V_list_X2=Rota_point_t[:,:,[1,2,0]]
   # V_list_X2 = torch.Tensor(Rota_point_t[:, :, 1], Rota_point_t[:, :, 2],Rota_point_t[:, :, 0])
   # V_list_X2=torch.cat([V_list_X2,Rota_point_t[:, :, 0]],dim=2)
    #print(K_list_view.size())
    #print(Rota_point_t_view.size())
    k_p_v=torch.matmul(torch.transpose(K_list_view,3,2).cuda(),Rota_point_t_view.cuda())

    #print(Cos_theat.size())
   # print(K_list.size())
    first_term=Cos_theat*Rota_point_t
   # print("first term is",first_term.size())
    K_list=K_list.view(K_list.size(0),-1,3)
    second_term=(1-Cos_theat)*(k_p_v.view(k_p_v.size(0),-1,1)*K_list)
   # print("second term is", second_term.size())
    third_term=Sin_theat*(K_list_X1*V_list_X1-K_list_X2*V_list_X2)
   # print("third term is", third_term.size())
    Rota_point_t2=first_term+second_term+third_term

    pointsReconstructed = Point_S * (Rota_point_t2+Point_T)
   # print(type(pointsReconstructed))
  #  print(pointsReconstructed.size())
    #end = time.time()
    #print("poster network time is ", end - start)
    return pointsReconstructed


def load_template():
    if not os.path.exists("./data/template/template.ply"):
        os.system("chmod +x ./data/download_template.sh")
        os.system("./data/download_template.sh")

    mesh = trimesh.load("./data/template/template.ply", process=False)
    point_set = mesh.vertices
    point_set, _, _ = pointcloud_processor.center_bounding_box(point_set)
    #print("vertex size is ", point_set.size())
    mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
    point_set_HR = mesh_HR.vertices
    point_set_HR, _, _ = pointcloud_processor.center_bounding_box(point_set_HR)

    vertex = torch.from_numpy(point_set).float()

    return vertex

# ===================CREATE optimizer================================= #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)

with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')
# ========================================================== #

# =============start of the learning loop ======================================== #
for epoch in range(opt.nepoch):
    if epoch==80:
        lrate = lrate/10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network.parameters(), lr=lrate)
    if epoch==90:
        lrate = lrate/10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network.parameters(), lr=lrate)

    # TRAIN MODE
    train_loss_L2_smpl.reset()
    network.train()
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        points, idx,_ , _= data
        points = points.transpose(2, 1).contiguous()
        points = points.cuda()
        pointsReconstructed =Tran_points(points, idx)#.forward_idx(points, idx)  # forward pass
        loss_net = torch.mean(
                (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
        loss_net.backward()
        train_loss_L2_smpl.update(loss_net.item())
        optimizer.step()  # gradient update

        # VIZUALIZE
        """
        if i % 100 == 0:
            vis.scatter(X=points.transpose(2, 1).contiguous()[0].data.cpu(),
            win = 'Train_input',
            opts = dict(
                title="Train_input",
                markersize=2,
            ),
            )
            vis.scatter(X=pointsReconstructed[0].data.cpu(),
            win = 'Train_output',
            opts = dict(
                title="Train_output",
                markersize=2,
            ),
            )
        """
        print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32,  loss_net.item()))

    # Validation
    with torch.no_grad():
        #val on SMPL data
        network.eval()
        val_loss_L2_smpl.reset()
        for i, data in enumerate(dataloader_smpl_test, 0):
            points, fn, idx, _ = data
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            #pointsReconstructed = network(points)  # forward pass
            pointsReconstructed = Tran_points(points,idx)
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


        # UPDATE CURVES
        L2curve_train_smpl.append(train_loss_L2_smpl.avg)
        L2curve_val_smlp.append(val_loss_L2_smpl.avg)
        """
        vis.line(X=np.column_stack((np.arange(len(L2curve_train_smpl)), np.arange(len(L2curve_val_smlp)))),
                 Y=np.column_stack((np.array(L2curve_train_smpl), np.array(L2curve_val_smlp))),
                 win='loss',
                 opts=dict(title="loss", legend=["L2curve_train_smpl" + opt.env,"L2curve_val_smlp" + opt.env,]))

        vis.line(X=np.column_stack((np.arange(len(L2curve_train_smpl)), np.arange(len(L2curve_val_smlp)))),
                 Y=np.log(np.column_stack((np.array(L2curve_train_smpl), np.array(L2curve_val_smlp)))),
                 win='log',
                 opts=dict(title="log", legend=["L2curve_train_smpl" + opt.env,"L2curve_val_smlp" + opt.env,]))
        """
        # dump stats in log file
        log_table = {
            "val_loss_L2_smpl": val_loss_L2_smpl.avg,
            "train_loss_L2_smpl": train_loss_L2_smpl.avg,
            "epoch": epoch,
            "lr": lrate,
            "env": opt.env,
        }
        print(log_table)
        with open(logname, 'a') as f:  # open and append
            f.write('json_stats: ' + json.dumps(log_table) + '\n')
        #save latest network
        torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))
