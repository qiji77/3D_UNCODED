from __future__ import print_function
import sys

sys.path.append('./auxiliary/')
sys.path.append('./')
import my_utils
from torch.autograd import Variable
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import model
import ply
import time
from sklearn.neighbors import NearestNeighbors

sys.path.append("./extension/")
sys.path.append("/app/python/")
import dist_chamfer as ext

distChamfer = ext.chamferDist()
import trimesh
import torch
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
import pointcloud_processor

class Inference(object):
    def __init__(self, HR=0, nepoch=3000, model_path='trained_models/sup_human_network_last.pth', num_points=6890,
                 num_angles=100, clean=1, scale=1, project_on_target=0, save_path=None, LR_input=True):
        self.LR_input = LR_input
        self.HR = HR
        self.nepoch = nepoch
        self.model_path = model_path
        self.num_points = num_points
        self.num_angles = num_angles
        self.clean = clean
        self.scale = scale
        self.project_on_target = project_on_target
        self.distChamfer = ext.chamferDist()

        # load network
        self.network = model.AE_AtlasNet_Humans(num_points=self.num_points)
        self.network.cuda()
        self.network.apply(my_utils.weights_init)
        if self.model_path != '':
            print("Reload weights from : ", self.model_path)
            self.network.load_state_dict(torch.load(self.model_path))
        self.network.eval()

        self.neigh = NearestNeighbors(1, 0.4)
        self.mesh_ref = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_ref_LR = trimesh.load("./data/template/template.ply", process=False)

        # load colors
        self.red_LR = np.load("./data/template/red_LR.npy").astype("uint8")
        self.green_LR = np.load("./data/template/green_LR.npy").astype("uint8")
        self.blue_LR = np.load("./data/template/blue_LR.npy").astype("uint8")
        self.red_HR = np.load("./data/template/red_HR.npy").astype("uint8")
        self.green_HR = np.load("./data/template/green_HR.npy").astype("uint8")
        self.blue_HR = np.load("./data/template/blue_HR.npy").astype("uint8")
        self.save_path ="./results_HR_truth" #save_path
    def load_template(self):
        if not os.path.exists("./data/template/template.ply"):
            os.system("chmod +x ./data/download_template.sh")
            os.system("./data/download_template.sh")

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        point_set = mesh.vertices
        point_set, _, _ = pointcloud_processor.center_bounding_box(point_set)

        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set_HR = mesh_HR.vertices
        point_set_HR, _, _ = pointcloud_processor.center_bounding_box(point_set_HR)

        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        self.prop = pointcloud_processor.get_vertex_normalised_area(mesh)
        assert (np.abs(np.sum(self.prop) - 1) < 0.001), "Propabilities do not sum to 1!)"
        self.prop = torch.from_numpy(self.prop).cuda().unsqueeze(0).float()
    def Tran_points_test(self, input):
        #  start=time.time()
        pointtran = self.network(input)
        # print("pointtran test is",pointtran)
        #  end=time.time()
        # print("network time is ",end-start)
        start = time.time()
        # pointtran_t=pointtran.view(self.points.size(0), -1, 10)
        # print(pointtran_t.size())
        # indices_s = torch.tensor([0,1,2]).cuda()
        # Point_S = torch.index_select(pointtran, dim=2, index=indices_s)

        indices_s = torch.tensor([0]).cuda()
        Point_S = torch.index_select(pointtran, dim=2, index=indices_s)
        indices_r = torch.tensor([1, 2, 3]).cuda()
        Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
        indices_theat = torch.tensor([4]).cuda()
        Point_theat = torch.index_select(pointtran, dim=2, index=indices_theat)
        indices_t = torch.tensor([5, 6, 7]).cuda()
        Point_T = torch.index_select(pointtran, dim=2, index=indices_t)

        rand_grid = self.vertex  # 6890, 3
        rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0).expand(input.size(0), 3, -1)
        # print("rand_grid test is",rand_grid)
        Rota_point = rand_grid
        Rota_point_t = Rota_point.transpose(2, 1).cuda()
        Rota_point_t2 = torch.zeros(Rota_point_t.size(0), Rota_point_t.size(1), Rota_point_t.size(2)).cuda()
        Cos_theat = torch.cos(Point_theat)
        point_r_list = Point_R.view(Point_R.size(0), -1, 3, 1)
        K_list = point_r_list / torch.sqrt(torch.matmul(torch.transpose(point_r_list, 3, 2), point_r_list))
        K_list_view = K_list.view(K_list.size(0), -1, 3, 1)
        Rota_point_t_view = Rota_point_t.view(Rota_point_t.size(0), -1, 3, 1)
        # print(K_list.size())
        Sin_theat = torch.sin(Point_theat)
        K_list_X1 = torch.cat([K_list[:, :, 1], K_list[:, :, 2]], dim=2)
        K_list_X1 = torch.cat([K_list_X1, K_list[:, :, 0]], dim=2)
        # K_list_X1
        # print("K_list_X1 :",K_list_X1.size())
        K_list_X2 = torch.cat([K_list[:, :, 2], K_list[:, :, 0]], dim=2)
        K_list_X2 = torch.cat([K_list_X2, K_list[:, :, 1]], dim=2)
        V_list_X1 = Rota_point_t[:, :, [2, 0, 1]]
        # V_list_X1 = torch.Tensor([Rota_point_t[:, :, 2], Rota_point_t[:, :, 0]],Rota_point_t[:, :, 1])
        # V_list_X1=torch.cat([V_list_X1,Rota_point_t[:, :, 1]],dim=2)
        V_list_X2 = Rota_point_t[:, :, [1, 2, 0]]
        # V_list_X2 = torch.Tensor(Rota_point_t[:, :, 1], Rota_point_t[:, :, 2],Rota_point_t[:, :, 0])
        # V_list_X2=torch.cat([V_list_X2,Rota_point_t[:, :, 0]],dim=2)
        # print(K_list_view.size())
        # print(Rota_point_t_view.size())
        k_p_v = torch.matmul(torch.transpose(K_list_view, 3, 2), Rota_point_t_view)

        # print(Cos_theat.size())
        # print(K_list.size())
        first_term = Cos_theat * Rota_point_t
        # print("first term is",first_term.size())
        K_list = K_list.view(K_list.size(0), -1, 3)
        second_term = (1 - Cos_theat) * (k_p_v.view(k_p_v.size(0), -1, 1) * K_list)
        # print("second term is", second_term.size())
        third_term = Sin_theat * (K_list_X1 * V_list_X1 - K_list_X2 * V_list_X2)
        # print("third term is", third_term.size())
        Rota_point_t2 = first_term + second_term + third_term

        pointsReconstructed = Point_S * (
                    Rota_point_t2 + Point_T)  # (Rota_point_t2+Point_T) #Point_S * (Rota_point_t2+Point_T)
        # print(type(pointsReconstructed))
        #  print(pointsReconstructed.size())
        # end = time.time()
        # print("poster network time is ", end - start)
        return pointsReconstructed
    def Tran_points_decode_full(self, input_data):
        #  start=time.time()
        pointtran = self.network.decode_full(input_data)
        # print("pointtran test is",pointtran)
        #  end=time.time()
        # print("network time is ",end-start)
        start = time.time()
        # pointtran_t=pointtran.view(self.points.size(0), -1, 10)
        # print(pointtran_t.size())
        # indices_s = torch.tensor([0,1,2]).cuda()
        # Point_S = torch.index_select(pointtran, dim=2, index=indices_s)

        indices_s = torch.tensor([0]).cuda()
        Point_S = torch.index_select(pointtran, dim=2, index=indices_s)
        indices_r = torch.tensor([1, 2, 3]).cuda()
        Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
        indices_theat = torch.tensor([4]).cuda()
        Point_theat = torch.index_select(pointtran, dim=2, index=indices_theat)
        indices_t = torch.tensor([5, 6, 7]).cuda()
        Point_T = torch.index_select(pointtran, dim=2, index=indices_t)
        div=20
        print(self.num_vertex_HR)
        print(self.vertex_HR.size())
        #batch = int(self.num_vertex_HR / div)
        #print(batch)
        i=div-1

        rand_grid = self.vertex_HR.view(input_data.size(0), -1, 3).transpose(1, 2).contiguous()
        rand_grid = Variable(rand_grid)
        print(rand_grid.size())
        #rand_grid = self.vertex  # 6890, 3
        #rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0).expand(input_data.size(0), 3, -1)
        # print("rand_grid test is",rand_grid)
        Rota_point = rand_grid
        Rota_point_t = Rota_point.transpose(2, 1).cuda()
        Rota_point_t2 = torch.zeros(Rota_point_t.size(0), Rota_point_t.size(1), Rota_point_t.size(2)).cuda()
        Cos_theat = torch.cos(Point_theat)
        point_r_list = Point_R.view(Point_R.size(0), -1, 3, 1)
        K_list = point_r_list / torch.sqrt(torch.matmul(torch.transpose(point_r_list, 3, 2), point_r_list))
        K_list_view = K_list.view(K_list.size(0), -1, 3, 1)
        Rota_point_t_view = Rota_point_t.view(Rota_point_t.size(0), -1, 3, 1)
        # print(K_list.size())
        Sin_theat = torch.sin(Point_theat)
        K_list_X1 = torch.cat([K_list[:, :, 1], K_list[:, :, 2]], dim=2)
        K_list_X1 = torch.cat([K_list_X1, K_list[:, :, 0]], dim=2)
        # K_list_X1
        # print("K_list_X1 :",K_list_X1.size())
        K_list_X2 = torch.cat([K_list[:, :, 2], K_list[:, :, 0]], dim=2)
        K_list_X2 = torch.cat([K_list_X2, K_list[:, :, 1]], dim=2)
        V_list_X1 = Rota_point_t[:, :, [2, 0, 1]]
        # V_list_X1 = torch.Tensor([Rota_point_t[:, :, 2], Rota_point_t[:, :, 0]],Rota_point_t[:, :, 1])
        # V_list_X1=torch.cat([V_list_X1,Rota_point_t[:, :, 1]],dim=2)
        V_list_X2 = Rota_point_t[:, :, [1, 2, 0]]
        # V_list_X2 = torch.Tensor(Rota_point_t[:, :, 1], Rota_point_t[:, :, 2],Rota_point_t[:, :, 0])
        # V_list_X2=torch.cat([V_list_X2,Rota_point_t[:, :, 0]],dim=2)
        # print(K_list_view.size())
        # print(Rota_point_t_view.size())
        k_p_v = torch.matmul(torch.transpose(K_list_view, 3, 2), Rota_point_t_view)

        # print(Cos_theat.size())
        # print(K_list.size())
        first_term = Cos_theat * Rota_point_t
        # print("first term is",first_term.size())
        K_list = K_list.view(K_list.size(0), -1, 3)
        second_term = (1 - Cos_theat) * (k_p_v.view(k_p_v.size(0), -1, 1) * K_list)
        third_term = Sin_theat * (K_list_X1 * V_list_X1 - K_list_X2 * V_list_X2)
        Rota_point_t2 = first_term + second_term + third_term
        pointsReconstructed = Point_S * (
                    Rota_point_t2 + Point_T)  # (Rota_point_t2+Point_T) #Point_S * (Rota_point_t2+Point_T)
        return pointsReconstructed
    def Tran_points_decode(self, input_data):
        #  start=time.time()
        pointtran = self.network.decode(input_data)
        # print("pointtran test is",pointtran)
        #  end=time.time()
        # print("network time is ",end-start)
        start = time.time()
        # pointtran_t=pointtran.view(self.points.size(0), -1, 10)
        # print(pointtran_t.size())
        # indices_s = torch.tensor([0,1,2]).cuda()
        # Point_S = torch.index_select(pointtran, dim=2, index=indices_s)

        indices_s = torch.tensor([0]).cuda()
        Point_S = torch.index_select(pointtran, dim=2, index=indices_s)
        indices_r = torch.tensor([1, 2, 3]).cuda()
        Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
        indices_theat = torch.tensor([4]).cuda()
        Point_theat = torch.index_select(pointtran, dim=2, index=indices_theat)
        indices_t = torch.tensor([5, 6, 7]).cuda()
        Point_T = torch.index_select(pointtran, dim=2, index=indices_t)

        rand_grid = self.vertex  # 6890, 3
        rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0).expand(input_data.size(0), 3, -1)
        # print("rand_grid test is",rand_grid)
        Rota_point = rand_grid
        Rota_point_t = Rota_point.transpose(2, 1).cuda()
        Rota_point_t2 = torch.zeros(Rota_point_t.size(0), Rota_point_t.size(1), Rota_point_t.size(2)).cuda()
        Cos_theat = torch.cos(Point_theat)
        point_r_list = Point_R.view(Point_R.size(0), -1, 3, 1)
        K_list = point_r_list / torch.sqrt(torch.matmul(torch.transpose(point_r_list, 3, 2), point_r_list))
        K_list_view = K_list.view(K_list.size(0), -1, 3, 1)
        Rota_point_t_view = Rota_point_t.view(Rota_point_t.size(0), -1, 3, 1)
        # print(K_list.size())
        Sin_theat = torch.sin(Point_theat)
        K_list_X1 = torch.cat([K_list[:, :, 1], K_list[:, :, 2]], dim=2)
        K_list_X1 = torch.cat([K_list_X1, K_list[:, :, 0]], dim=2)
        # K_list_X1
        # print("K_list_X1 :",K_list_X1.size())
        K_list_X2 = torch.cat([K_list[:, :, 2], K_list[:, :, 0]], dim=2)
        K_list_X2 = torch.cat([K_list_X2, K_list[:, :, 1]], dim=2)
        V_list_X1 = Rota_point_t[:, :, [2, 0, 1]]
        # V_list_X1 = torch.Tensor([Rota_point_t[:, :, 2], Rota_point_t[:, :, 0]],Rota_point_t[:, :, 1])
        # V_list_X1=torch.cat([V_list_X1,Rota_point_t[:, :, 1]],dim=2)
        V_list_X2 = Rota_point_t[:, :, [1, 2, 0]]
        # V_list_X2 = torch.Tensor(Rota_point_t[:, :, 1], Rota_point_t[:, :, 2],Rota_point_t[:, :, 0])
        # V_list_X2=torch.cat([V_list_X2,Rota_point_t[:, :, 0]],dim=2)
        # print(K_list_view.size())
        # print(Rota_point_t_view.size())
        k_p_v = torch.matmul(torch.transpose(K_list_view, 3, 2), Rota_point_t_view)

        # print(Cos_theat.size())
        # print(K_list.size())
        first_term = Cos_theat * Rota_point_t
        # print("first term is",first_term.size())
        K_list = K_list.view(K_list.size(0), -1, 3)
        second_term = (1 - Cos_theat) * (k_p_v.view(k_p_v.size(0), -1, 1) * K_list)
        third_term = Sin_theat * (K_list_X1 * V_list_X1 - K_list_X2 * V_list_X2)
        Rota_point_t2 = first_term + second_term + third_term
        pointsReconstructed = Point_S * (
                    Rota_point_t2 + Point_T)  # (Rota_point_t2+Point_T) #Point_S * (Rota_point_t2+Point_T)
        return pointsReconstructed
    def compute_correspondances(self, source_p, source_reconstructed_p, target_p, target_reconstructed_p, path):
        """
        Given 2 meshes, and their reconstruction, compute correspondences between the 2 meshes through neireast neighbors
        :param source_p: path for source mesh
        :param source_reconstructed_p: path for source mesh reconstructed
        :param target_p: path for target mesh
        :param target_reconstructed_p: path for target mesh reconstructed
        :return: None but save a file with correspondences
        """
        # inputs are all filepaths
        with torch.no_grad():
            source = trimesh.load(source_p, process=False)
            source_reconstructed = trimesh.load(source_reconstructed_p, process=False)
            target = trimesh.load(target_p, process=False)
            target_reconstructed = trimesh.load(target_reconstructed_p, process=False)

            # project on source_reconstructed
            self.neigh.fit(source_reconstructed.vertices)
            idx_knn = self.neigh.kneighbors(source.vertices, return_distance=False)

            # correspondances throught template
            closest_points = target_reconstructed.vertices[idx_knn]
            print("before np.mean",closest_points.size)
            closest_points = np.mean(closest_points, 1, keepdims=False)
            print("After np.mean", closest_points.size)
            # project on target
            if self.project_on_target:
                print("projection on target...")
                self.neigh.fit(target.vertices)
                idx_knn = self.neigh.kneighbors(closest_points, return_distance=False)
                closest_points = target.vertices[idx_knn]
                closest_points = np.mean(closest_points, 1, keepdims=False)

            # save output
            if path is None:
                np.savetxt("results_HR_truth/correspondences.txt", closest_points, fmt='%1.10f')
            else:
                print("Path is",path)
                np.savetxt(os.path.join(self.save_path, path), closest_points, fmt='%1.10f')
            mesh = trimesh.Trimesh(vertices=closest_points, faces=source.faces, process=False)
            mesh.export("results_HR_truth/correspondences.ply")

    def forward(self, inputA="data/example_0.ply", inputB="data/example_1.ply", path=None):
        print("computing correspondences for " + inputA + " and " + inputB)
        start = time.time()
        with open("data/inter_challenge.txt", "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                inputA = "data/scans/test_scan_" + line[:3] + ".ply"
                inputB = "data/scans/test_scan_" + line[4:7] + ".ply"
                # Reconstruct meshes
                self.reconstruct(inputA)
                self.reconstruct(inputB)

                # Compute the correspondences through the recontruction
                if self.save_path is None:
                    self.compute_correspondances(inputA, inputA[:-4] + "FinalReconstruction.ply", inputB,
                                                 inputB[:-4] + "FinalReconstruction.ply",
                                                 "results_HR_truth/" + line[:3] + "_" + line[4:7] + ".txt")
                else:
                    self.compute_correspondances(inputA,
                                                 os.path.join(self.save_path,
                                                              inputA[-8:-4] + "FinalReconstruction.ply"),
                                                 inputB,
                                                 os.path.join(self.save_path,
                                                              inputB[-8:-4] + "FinalReconstruction.ply"),
                                                 "results_HR_truth/" +line[:3] + "_" + line[4:7] + ".txt")

                end = time.time()
                print("ellapsed time is ", end - start, " seconds !")

    def regress(self, points):
        """
        search the latent space to global_variables. Optimize reconstruction using the Chamfer Distance
        :param points: input points to reconstruct
        :return pointsReconstructed: final reconstruction after optimisation
        """
        points = points.data
        latent_code = self.network.encoder(points)
        lrate = 0.001  # learning rate
        # define parameters to be optimised and optimiser
        input_param = nn.Parameter(latent_code.data, requires_grad=True)
        self.optimizer = optim.Adam([input_param], lr=lrate)
        loss = 10
        i = 0
        #import IPython;IPython.embed()
        # learning loop
        while np.log(loss) > -9 and i < self.nepoch:
            self.optimizer.zero_grad()
            pointsReconstructed =self.Tran_points_decode(input_param)#self.network.decode(input_param) #self.Tran_points_decode(input_param)  # forward pass
            dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
            loss_net.backward()
            self.optimizer.step()
            loss = loss_net.item()
            i = i + 1
        with torch.no_grad():
            if self.HR:
                pointsReconstructed =self.Tran_points_decode_full(input_param)  # forward pass#self.network.decode_full(input_param)
            else:
                pointsReconstructed =self.Tran_points_decode(input_param)  # forward pass self.network.decode(input_param)#self.Tran_points_decode(input_param)  # forward pass

        print("loss reg : ", loss)
        return pointsReconstructed

    def run(self, input, scalefactor, path):
        """
        :param input: input mesh to reconstruct optimally.
        :return: final reconstruction after optimisation
        """
        self.load_template()
        input, translation = my_utils.center(input)
        if not self.HR:
            mesh_ref = self.mesh_ref_LR
        else:
            mesh_ref = self.mesh_ref

        ## Extract points and put them on GPU
        points = input.vertices
        # TODO : remove random here
        random_sample = np.random.choice(np.shape(points)[0], size=10000)

        points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
        points = points.transpose(2, 1).contiguous()
        points = points.cuda()
        #print("Points size is ",points.size())
        # Get a low resolution PC to find the best reconstruction after a rotation on the Y axis
        if self.LR_input:
            print("Using a Low_res input")
            points_LR = torch.from_numpy(input.vertices[random_sample].astype(np.float32)).contiguous().unsqueeze(0)
        else:
            print("Using a High_res input")
            points_LR = torch.from_numpy(input.vertices.astype(np.float32)).contiguous().unsqueeze(0)

        input_LR_mesh = trimesh.Trimesh(vertices=(points_LR.squeeze().data.cpu().numpy() + translation) / scalefactor,
                                        faces=np.array([1, 2, 3]), process=False)
        if self.save_path is None:
            input_LR_mesh.export(path[:-4] + "DownsampledInput.ply")
        else:
            input_LR_mesh.export(os.path.join(self.save_path, path[-8:-4] + "DownsampledInput.ply"))

        points_LR = points_LR.transpose(2, 1).contiguous()
        points_LR = points_LR.cuda()
        #print("points_LR size is ", points_LR.size())
        theta = 0
        bestLoss = 1000
        pointsReconstructed =self.Tran_points_test(points_LR)
        dist1, dist2 = distChamfer(points_LR.transpose(2, 1).contiguous(), pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        print("loss without rotation: ", loss_net.item(),
              0)  # ---- Search best angle for best reconstruction on the Y axis---

        x = np.linspace(-np.pi / 2, np.pi / 2, self.num_angles)
        y = np.linspace(-np.pi / 4, np.pi / 4, self.num_angles // 4)

        THETA, PHI = np.meshgrid(x, y)
        Z = np.ndarray([THETA.shape[0], THETA.shape[1]])
        for j in range(THETA.shape[1]):
            for i in range(THETA.shape[0]):
                if self.num_angles == 1:
                    theta = 0
                    phi = 0
                theta = THETA[i, j]
                phi = PHI[i, j]

                #  Rotate mesh by theta and renormalise
                rot_matrix = np.array(
                    [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [- np.sin(theta), 0, np.cos(theta)]])
                rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
                rot_matrix = torch.matmul(torch.from_numpy(np.array(
                    [[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1], ])).float().cuda(),
                                          rot_matrix)
                points2 = torch.matmul(rot_matrix, points_LR)
                mesh_tmp = trimesh.Trimesh(process=False, use_embree=False,
                                           vertices=points2[0].transpose(1, 0).data.cpu().numpy(),
                                           faces=self.network.mesh.faces)
                # bbox
                bbox = np.array([[np.max(mesh_tmp.vertices[:, 0]), np.max(mesh_tmp.vertices[:, 1]),
                                  np.max(mesh_tmp.vertices[:, 2])],
                                 [np.min(mesh_tmp.vertices[:, 0]), np.min(mesh_tmp.vertices[:, 1]),
                                  np.min(mesh_tmp.vertices[:, 2])]])
                norma = torch.from_numpy((bbox[0] + bbox[1]) / 2).float().cuda()

                norma2 = norma.unsqueeze(1).expand(3, points2.size(2)).contiguous()
                points2[0] = points2[0] - norma2
                #from IPython import embed
                # reconstruct rotated mesh
                pointsReconstructed = self.Tran_points_test(points2)#self.network(points2)
                dist1, dist2 = distChamfer(points2.transpose(2, 1).contiguous(), pointsReconstructed)

                loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
                Z[i, j] = loss_net.item()
                if loss_net < bestLoss:
                    print(theta, phi, loss_net)
                    bestLoss = loss_net
                    best_theta = theta
                    best_phi = phi
                    # unrotate the mesh
                    norma3 = norma.unsqueeze(0).expand(pointsReconstructed.size(1), 3).contiguous()
                    pointsReconstructed[0] = pointsReconstructed[0] + norma3
                    rot_matrix = np.array(
                        [[np.cos(-theta), 0, np.sin(-theta)], [0, 1, 0], [- np.sin(-theta), 0, np.cos(-theta)]])
                    rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
                    rot_matrix = torch.matmul(rot_matrix, torch.from_numpy(np.array(
                        [[np.cos(-phi), np.sin(-phi), 0], [-np.sin(-phi), np.cos(-phi), 0],
                         [0, 0, 1], ])).float().cuda())
                    pointsReconstructed = torch.matmul(pointsReconstructed, rot_matrix.transpose(1, 0))
                    bestPoints = pointsReconstructed

        try:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(THETA, PHI, -Z, rstride=1, cstride=1,
                            cmap='magma', edgecolor='none', alpha=0.8)

            ax.set_xlabel('THETA', fontsize=20)
            ax.set_ylabel('PHI', fontsize=20)
            ax.set_zlabel('CHAMFER', fontsize=20)
            ax.scatter(best_theta, best_phi, -bestLoss.item(), marker='*', c="red", s=100, alpha=1)
            ax.scatter(best_theta, best_phi, np.min(-Z), marker='*', c="red", s=100, alpha=1)
            ax.view_init(elev=45., azim=45)
            plt.savefig("3Dcurve.png")
            if self.save_path is not None:
                plt.savefig(os.path.join(self.save_path, path[-8:-4] + "3Dcurve.png"))
            else:
                plt.savefig(path[:-4] + "3Dcurve.png")

        except:
            pass
        # for theta in np.linspace(-np.pi/2, np.pi/2, self.num_angles):
        #     if self.num_angles == 1:
        #         theta = 0
        #     X.append(theta)
        #
        #     #  Rotate mesh by theta and renormalise
        #     rot_matrix = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [- np.sin(theta), 0,  np.cos(theta)]])
        #     rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
        #     points2 = torch.matmul(rot_matrix, points_LR).squeeze()
        #     #bbox
        #     bbox = torch.Tensor([[torch.max(points2[0]), torch.max(points2[1]), torch.max(points2[2])], [torch.min(points2[0]), torch.min(points2[1]), torch.min(points2[2])]])
        #     norma = ((bbox[0] + bbox[1]) / 2).float().cuda()
        #     norma = norma.cuda()
        #     points2 = points2.unsqueeze(0)
        #     norma2 = norma.unsqueeze(1).expand(3,points2.size(2)).contiguous()
        #     points2[0] = points2[0] - norma2
        #
        #     # reconstruct rotated mesh
        #     pointsReconstructed = self.network(points2)
        #     dist1, dist2 = distChamfer(points2.transpose(2, 1).contiguous(), pointsReconstructed)
        #
        #
        #     loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        #     Y.append(loss_net.item())
        #     if loss_net < bestLoss:
        #         bestLoss = loss_net
        #         best_theta = theta
        #         # unrotate the mesh
        #         norma3 = norma.unsqueeze(0).expand(pointsReconstructed.size(1), 3).contiguous()
        #         pointsReconstructed[0] = pointsReconstructed[0] + norma3
        #         rot_matrix = np.array([[np.cos(-theta), 0, np.sin(-theta)], [0, 1, 0], [- np.sin(-theta), 0,  np.cos(-theta)]])
        #         rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
        #         pointsReconstructed = torch.matmul(pointsReconstructed, rot_matrix.transpose(1,0))
        #         bestPoints = pointsReconstructed

        print("best loss and theta and phi : ", bestLoss.item(), best_theta, best_phi)

        if self.HR:
            faces_tosave = self.network.mesh_HR.faces
        else:
            faces_tosave = self.network.mesh.faces

        # create initial guess
        mesh = trimesh.Trimesh(vertices=(bestPoints[0].data.cpu().numpy() + translation) / scalefactor,
                               faces=self.network.mesh.faces, process=False)
        try:
            #plt.plot(X, Y)
            plt.savefig("curve.png")
        except:
            pass
        # START REGRESSION
        print("start regression...")

        # rotate with optimal angle
        rot_matrix = np.array(
            [[np.cos(best_theta), 0, np.sin(best_theta)], [0, 1, 0], [- np.sin(best_theta), 0, np.cos(best_theta)]])
        rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
        points2 = torch.matmul(rot_matrix, points)
        mesh_tmp = trimesh.Trimesh(vertices=points2[0].transpose(1, 0).data.cpu().numpy(),
                                   faces=self.network.mesh.faces, process=False)
        bbox = np.array(
            [[np.max(mesh_tmp.vertices[:, 0]), np.max(mesh_tmp.vertices[:, 1]), np.max(mesh_tmp.vertices[:, 2])],
             [np.min(mesh_tmp.vertices[:, 0]), np.min(mesh_tmp.vertices[:, 1]), np.min(mesh_tmp.vertices[:, 2])]])
        norma = torch.from_numpy((bbox[0] + bbox[1]) / 2).float().cuda()
        norma2 = norma.unsqueeze(1).expand(3, points2.size(2)).contiguous()
        points2[0] = points2[0] - norma2
        pointsReconstructed1 = self.regress(points2)
        # unrotate with optimal angle
        norma3 = norma.unsqueeze(0).expand(pointsReconstructed1.size(1), 3).contiguous()
        rot_matrix = np.array(
            [[np.cos(-best_theta), 0, np.sin(-best_theta)], [0, 1, 0], [- np.sin(-best_theta), 0, np.cos(-best_theta)]])
        rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
        pointsReconstructed1[0] = pointsReconstructed1[0] + norma3
        pointsReconstructed1 = torch.matmul(pointsReconstructed1, rot_matrix.transpose(1, 0))

        # create optimal reconstruction
        meshReg = trimesh.Trimesh(vertices=(pointsReconstructed1[0].data.cpu().numpy() + translation) / scalefactor,
                                  faces=faces_tosave, process=False)

        print("... Done!")
        return mesh, meshReg

    def save(self, mesh, mesh_color, path, red, green, blue):
        """
        Home-made function to save a ply file with colors. A bit hacky
        """
        to_write = mesh.vertices
        b = np.zeros((len(mesh.faces), 4)) + 3
        b[:, 1:] = np.array(mesh.faces)
        try:
            points2write = pd.DataFrame({
                'lst0Tite': to_write[:, 0],
                'lst1Tite': to_write[:, 1],
                'lst2Tite': to_write[:, 2],
                'lst3Tite': red,
                'lst4Tite': green,
                'lst5Tite': blue,
            })
            ply.write_ply(filename=path, points=points2write, as_text=True, text=False,
                          faces=pd.DataFrame(b.astype(int)),
                          color=True)
        except:
            points2write = pd.DataFrame({
                'lst0Tite': to_write[:, 0],
                'lst1Tite': to_write[:, 1],
                'lst2Tite': to_write[:, 2],
            })
            ply.write_ply(filename=path, points=points2write, as_text=True, text=False,
                          faces=pd.DataFrame(b.astype(int)),
                          color=False)

    def reconstruct(self, input_p):
        """
        Recontruct a 3D shape by deforming a template
        :param input_p: input path
        :return: None (but save reconstruction)
        """
        print("Reconstructing ", input_p)
        input = trimesh.load(input_p, process=False)
        scalefactor = 1.0
        if self.scale:
            input, scalefactor = my_utils.scale(input,
                                                self.mesh_ref_LR)  # scale input to have the same volume as mesh_ref_LR
        if self.clean:
            input = my_utils.clean(input)  # remove points that doesn't belong to any edges
        my_utils.test_orientation(input)
        mesh, meshReg = self.run(input, scalefactor, input_p)

        if not self.HR:
            red = self.red_LR
            green = self.green_LR
            blue = self.blue_LR
            mesh_ref = self.mesh_ref_LR
        else:
            blue = self.blue_HR
            red = self.red_HR
            green = self.green_HR
            mesh_ref = self.mesh_ref

        if self.save_path is None:
            self.save(mesh, self.mesh_ref_LR, input_p[:-4] + "InitialGuess.ply", self.red_LR, self.green_LR,
                      self.blue_LR)
            self.save(meshReg, mesh_ref, input_p[:-4] + "FinalReconstruction.ply", red, green, blue)
        else:
            self.save(mesh, self.mesh_ref_LR, os.path.join(self.save_path, input_p[-8:-4] + "InitialGuess.ply"),
                      self.red_LR, self.green_LR, self.blue_LR)
            self.save(meshReg, mesh_ref, os.path.join(self.save_path, input_p[-8:-4] + "FinalReconstruction.ply"), red,
                      green, blue)

        # Save optimal reconstruction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR', type=int, default=1,
                        help='Use high Resolution template for better precision in the nearest neighbor step ?')
    parser.add_argument('--LR_input', type=int, default=1,
                        help='Use high Resolution template for better precision in the nearest neighbor step ?')
    parser.add_argument('--nepoch', type=int, default=3000,
                        help='number of epochs to train for during the regression step')
    parser.add_argument('--model_path', type=str, default='trained_models/sup_human_network_last.pth',
                        help='your path to the trained model')
    parser.add_argument('--inputA', type=str, default="data/example_0.ply", help='your path to mesh 0')
    parser.add_argument('--inputB', type=str, default="data/example_1.ply", help='your path to mesh 1')
    parser.add_argument('--num_points', type=int, default=6890, help='number of points fed to poitnet')
    parser.add_argument('--num_angles', type=int, default=100,
                        help='number of angle in the search of optimal reconstruction. Set to 1, if you mesh are already facing the cannonical direction as in data/example_1.ply')
    #parser.add_argument('--env', type=str, default="CODED", help='visdom environment')
    parser.add_argument('--clean', type=int, default=1, help='if 1, remove points that dont belong to any edges')
    parser.add_argument('--scale', type=int, default=1,
                        help='if 1, scale input mesh to have same volume as the template')
    parser.add_argument('--project_on_target', type=int, default=0,
                        help='if 1, projects predicted correspondences point on target mesh')
    parser.add_argument('--randomize', type=int, default=0,
                        help='if 1, projects predicted correspondences point on target mesh')

    opt = parser.parse_args()

    opt.HR = my_utils.int_2_boolean(opt.HR)
    opt.LR_input = my_utils.int_2_boolean(opt.LR_input)
    opt.clean = my_utils.int_2_boolean(opt.clean)
    opt.scale = my_utils.int_2_boolean(opt.scale)
    opt.project_on_target = my_utils.int_2_boolean(opt.project_on_target)
    opt.randomize = my_utils.int_2_boolean(opt.randomize)

    my_utils.plant_seeds(randomized_seed=opt.randomize)
    inf = Inference(HR=opt.HR, nepoch=opt.nepoch, model_path=opt.model_path, num_points=opt.num_points,
                    num_angles=opt.num_angles, clean=opt.clean, scale=opt.scale,
                    project_on_target=opt.project_on_target, LR_input=opt.LR_input)
    # inf.reconstruct((opt.inputA))
    inf.forward(opt.inputA, opt.inputB)
    # inf.forward(opt.inputA, opt.inputA)
