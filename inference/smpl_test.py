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
class SMPL_t(object):
    def __init__(self, HR=0, nepoch=3000, num_points=6890,
                 num_angles=100,save_path=None):
        self.save_path = save_path
        self.right_arm=np.load("./data/output/left_hand_indices_t.npy")
        self.red_LR = np.load("./data/template/red_LR.npy").astype("uint8")
        self.green_LR = np.load("./data/template/green_LR.npy").astype("uint8")
        self.blue_LR = np.load("./data/template/blue_LR.npy").astype("uint8")
        self.red_HR = np.load("./data/template/red_HR.npy").astype("uint8")
        self.green_HR = np.load("./data/template/green_HR.npy").astype("uint8")
        self.blue_HR = np.load("./data/template/blue_HR.npy").astype("uint8")
        self.mesh_ref_LR = trimesh.load("./data/template/template.ply", process=False)
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
        red = self.red_LR
        green = self.green_LR
        blue = self.blue_LR
        mesh_ref = self.mesh_ref_LR
        self.save(mesh,mesh_ref,path=self.save_path,red=red,green=green,blue=blue)
    def save(self, mesh, mesh_color, path, red, green, blue):
        """
        Home-made function to save a ply file with colors. A bit hacky
        """
        #for i in range(self.right_arm.size()):
        print(self.right_arm)
        mesh.vertices[self.right_arm]=[0,0,0]
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

if __name__ == '__main__':
    inf = SMPL_t(save_path="./result_smpl")
    inf.load_template()

