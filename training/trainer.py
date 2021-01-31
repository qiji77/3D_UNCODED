import torch
import torch.optim as optim
import time
import my_utils
import model
import extension.get_chamfer as get_chamfer
import dataset
import trimesh
import numpy as np
import pointcloud_processor
from termcolor import colored
from abstract_trainer import AbstractTrainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
class Trainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.git_repo_path = "https://github.com/ThibaultGROUEIX/3D-CODED/commit/"
        self.init_save_dict(opt)
        self.dataset_train = None

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        network = model.AE_AtlasNet_Humans(point_translation=self.opt.point_translation,
                                           dim_template=self.opt.dim_template,
                                           patch_deformation=self.opt.patch_deformation,
                                           dim_out_patch=self.opt.dim_out_patch,
                                           start_from=self.opt.start_from, dataset_train=self.dataset_train)
        network.cuda()  # put network on GPU
        network.apply(my_utils.weights_init)  # initialization of the weight
        if self.opt.model != "":
            try:
                network.load_state_dict(torch.load(self.opt.model))
                print(" Previous network weights loaded! From ", self.opt.model)
            except:
                print("Failed to reload ", self.opt.model)
        if self.opt.reload:
            print(f"reload model frow :  {self.opt.dir_name}/network.pth")
            self.opt.model = os.path.join(self.opt.dir_name, "network.pth")
            network.load_state_dict(torch.load(self.opt.model))

        self.network = network
        self.network.eval()#表示开始预测，.train()表示训练 主要区别是eval用到全连接，train用到部分权重
        self.network.save_template_png(self.opt.dir_name)
        # self.network.train()

    def build_optimizer(self):
        """
        Create optimizer
        """
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.opt.reload:
            self.optimizer.load_state_dict(torch.load(f'{self.opt.checkpointname}'))
            my_utils.yellow_print("Reloaded optimizer")

    def build_dataset_train(self):
        """
        Create training dataset
        """
        self.dataset_train = dataset.SURREAL(train=True, regular_sampling=True)
       # self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.opt.batch_size,
                                                           # shuffle=True, num_workers=int(self.opt.workers),
                                                           # drop_last=True)
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.opt.batch_size,
                                                            shuffle=True, num_workers=48,
                                                            drop_last=True)
        self.len_dataset = len(self.dataset_train)

    def build_dataset_test(self):
        """
        Create testing dataset
        """
        self.dataset_test = dataset.SURREAL(train=False)
        #self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=5,
                                                          # shuffle=False, num_workers=int(self.opt.workers),
                                                          # drop_last=True)
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=5,
                                                           shuffle=False, num_workers=48,
                                                           drop_last=True)
        self.len_dataset_test = len(self.dataset_test)

    def build_losses(self):
        """
        Create losses
        """
        self.distChamfer = get_chamfer.get(self.opt)
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
        #print(f"Using template to initialize template")
    def Tran_points_test(self):
      #  start=time.time()
        pointtran = self.network(self.points)
        #print("pointtran test is",pointtran)
      #  end=time.time()
       # print("network time is ",end-start)
        start = time.time()
        pointtran_t=pointtran.view(self.points.size(0), -1, 8)
        #print(pointtran_t.size())

        indices_s = torch.tensor([0]).cuda()
        Point_S = torch.index_select(pointtran, dim=2, index=indices_s)
        indices_r = torch.tensor([1, 2, 3]).cuda()
        Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
        indices_theat = torch.tensor([4]).cuda()
        Point_theat = torch.index_select(pointtran, dim=2, index=indices_theat)
        indices_t = torch.tensor([5, 6, 7]).cuda()
        Point_T = torch.index_select(pointtran, dim=2, index=indices_t)

        rand_grid = self.vertex  # 6890, 3
        rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0).expand(self.points.size(0), 3,-1)
        #print("rand_grid test is",rand_grid)
        Rota_point=rand_grid
        Rota_point_t = Rota_point.transpose(2, 1).cuda()
        Rota_point_t2=torch.zeros(Rota_point_t.size(0),Rota_point_t.size(1),Rota_point_t.size(2)).cuda()
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
       # print(K_list_view.size())
       # print(Rota_point_t_view.size())
        k_p_v=torch.matmul(torch.transpose(K_list_view,3,2),Rota_point_t_view)

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

        pointsReconstructed = Point_S*(Rota_point_t2+Point_T)
       # print(type(pointsReconstructed))
      #  print(pointsReconstructed.size())
        #end = time.time()
        #print("poster network time is ", end - start)
        return pointsReconstructed
    def Tran_points(self):
      #  start=time.time()
        point_clone=self.points
        print("before network self.points",point_clone)
        pointtran = self.network(self.points, self.idx)
        #("Trainer idx",self.idx)
        print("after network self.points",self.points)

        #print("diff is",diff)
        import IPython;IPython.embed()
      #  end=time.time()
       # print("network time is ",end-start)
        start = time.time()
        pointtran_t=pointtran.view(self.points.size(0), -1, 8)

        pointtran_tc=pointtran_t.clone()
        #print(pointtran_t.size())
        indices_s = torch.tensor([0]).cuda()
        Point_S = torch.index_select(pointtran, dim=2, index=indices_s)
        indices_r = torch.tensor([1,2,3]).cuda()
        Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
        indices_theat = torch.tensor([4]).cuda()
        Point_theat = torch.index_select(pointtran, dim=2, index=indices_theat)
        indices_t = torch.tensor([5,6,7]).cuda()
        Point_T = torch.index_select(pointtran, dim=2, index=indices_t)

        if not self.idx is None:
            idx = self.idx.view(-1)
            idx = idx.numpy().astype(np.int)

        rand_grid = self.vertex  # 6890, 3
        rand_grid = rand_grid[idx, :]  # batch x 2500, 3   扩展template
        rand_grid = rand_grid.view(self.points.size(0), -1, 3).transpose(2, 1).contiguous()  # batch , 2500, 3 -> batch, 6980, 3


        Rota_point = rand_grid #要旋转的点  也就是 基础模板点
        Rota_point_t = Rota_point.transpose(2, 1).cuda()
        #Rota_point_t2=torch.zeros(Rota_point_t.size(0),Rota_point_t.size(1),Rota_point_t.size(2)).cuda()
        Cos_theat=torch.cos(Point_theat)
        point_r_list=Point_R.view(Point_R.size(0),-1,3,1)#旋转轴
        K_list=torch.sqrt(torch.matmul(torch.transpose(point_r_list,3,2),point_r_list))*point_r_list
        K_list_view=K_list.view(K_list.size(0),-1,3,1)
        Rota_point_t_view=Rota_point_t.view(Rota_point_t.size(0),-1,3,1)
        #print(K_list.size())
        Sin_theat = torch.sin(Point_theat)
        K_list_X1=torch.cat([K_list[:,:,1], K_list[:,:,2]], dim=2)#计算叉乘
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
        k_p_v=torch.matmul(torch.transpose(K_list_view,3,2),Rota_point_t_view)

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

        pointsReconstructed =Point_S*(Rota_point_t2+Point_T)
       # print(type(pointsReconstructed))
      #  print(pointsReconstructed.size())
        #end = time.time()
        #print("poster network time is ", end - start)
        diff = torch.mean(
            (pointtran_tc - pointtran_t) ** 2)
        print("pointtran_tc diff is", diff)
        return pointsReconstructed,pointtran_t,idx

    def Reconstruction_loss(self,pointtran_t_input,idx):
        #  start=time.time()
        pointtran = pointtran_t_input
        #  end=time.time()
        # print("network time is ",end-start)
        start = time.time()
        pointtran_t = pointtran.view(self.points.size(0), -1, 8)
        # print(pointtran_t.size())
        indices_s = torch.tensor([0]).cuda()
        Point_S = torch.index_select(pointtran, dim=2, index=indices_s)
        indices_r = torch.tensor([1, 2, 3]).cuda()
        Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
        indices_theat = torch.tensor([4]).cuda()
        Point_theat = torch.index_select(pointtran, dim=2, index=indices_theat)
        indices_t = torch.tensor([5, 6, 7]).cuda()
        Point_T = torch.index_select(pointtran, dim=2, index=indices_t)

        #if not self.idx is None:
        #    idx = self.idx.view(-1)
        #    idx = idx.numpy().astype(np.int)

        rand_grid = self.vertex  # 6890, 3
        rand_grid = rand_grid[idx, :]  # batch x 2500, 3
        rand_grid = rand_grid.view(self.points.size(0), -1, 3).transpose(2,
                                                                         1).contiguous()  # batch , 2500, 3 -> batch, 6980, 3
        Rota_point = (self.points.transpose(2,1)/Point_S- Point_T)
        Rota_point_t = Rota_point.cuda()
        Rota_point_t2 = torch.zeros(Rota_point_t.size(0), Rota_point_t.size(1), Rota_point_t.size(2)).cuda()
        Cos_theat = torch.cos(-Point_theat)
        point_r_list = Point_R.view(Point_R.size(0), -1, 3, 1)
        K_list = torch.sqrt(torch.matmul(torch.transpose(point_r_list, 3, 2), point_r_list)) * point_r_list
        K_list_view = K_list.view(K_list.size(0), -1, 3, 1)
        Rota_point_t_view = Rota_point_t.view(Rota_point_t.size(0), -1, 3, 1)
        # print(K_list.size())
        Sin_theat = torch.sin(-Point_theat)
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

        pointsReconstructed = Rota_point_t2
        loss_train_total = torch.mean(
            (pointsReconstructed.view(self.points.size(0), -1, 3) - rand_grid.transpose(2, 1).contiguous()) ** 2)
        # print(type(pointsReconstructed))
        #  print(pointsReconstructed.size())
        # end = time.time()
        # print("poster network time is ", end - start)
        return loss_train_total
    def Reconstruction_loss_test(self):
        #  start=time.time()
        pointtran = self.network(self.points)
        #  end=time.time()
        # print("network time is ",end-start)
        start = time.time()
        pointtran_t = pointtran.view(self.points.size(0), -1, 8)
        # print(pointtran_t.size())
        indices_s = torch.tensor([0]).cuda()
        Point_S = torch.index_select(pointtran, dim=2, index=indices_s)
        indices_r = torch.tensor([1, 2, 3]).cuda()
        Point_R = torch.index_select(pointtran, dim=2, index=indices_r)
        indices_theat = torch.tensor([4]).cuda()
        Point_theat = torch.index_select(pointtran, dim=2, index=indices_theat)
        indices_t = torch.tensor([5, 6, 7]).cuda()
        Point_T = torch.index_select(pointtran, dim=2, index=indices_t)

        rand_grid = self.vertex  # 6890, 3
        rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0).expand(self.points.size(0), 3,
                                                                           -1)

        Rota_point = (self.points.transpose(2, 1) / Point_S - Point_T)
        Rota_point_t = Rota_point.cuda()
        Rota_point_t2 = torch.zeros(Rota_point_t.size(0), Rota_point_t.size(1), Rota_point_t.size(2)).cuda()
        Cos_theat = torch.cos(-Point_theat)
        point_r_list = Point_R.view(Point_R.size(0), -1, 3, 1)
        K_list = torch.sqrt(torch.matmul(torch.transpose(point_r_list, 3, 2), point_r_list)) * point_r_list
        K_list_view = K_list.view(K_list.size(0), -1, 3, 1)
        Rota_point_t_view = Rota_point_t.view(Rota_point_t.size(0), -1, 3, 1)
        # print(K_list.size())
        Sin_theat = torch.sin(-Point_theat)
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

        pointsReconstructed = Rota_point_t2
        loss_train_total = torch.mean(
            (pointsReconstructed.view(self.points.size(0), -1, 3) - rand_grid.transpose(2, 1).contiguous()) ** 2)
        # print(type(pointsReconstructed))
        #  print(pointsReconstructed.size())
        # end = time.time()
        # print("poster network time is ", end - start)
        return loss_train_total
    def train_iteration(self,epoch):
        self.optimizer.zero_grad()#清空所有被优化过的Variable的梯度.
        self.load_template()
        #pointsReconstructed = self.network(self.points, self.idx)  # forward pass # batch, num_point, 3

        pointsReconstructed,pointtran_t,idx=self.Tran_points()  # forward pass # batch, num_point, 3
       # start=time.time()
        if epoch>20:
            reloss = self.Reconstruction_loss(pointtran_t, idx)
            loss_train_total = torch.mean(
                    (pointsReconstructed.view(self.points.size(0), -1, 3) - self.points.transpose(2,
                                                                                                  1).contiguous()) ** 2) + 5*reloss
        else:
            loss_train_total = torch.mean(
                (pointsReconstructed.view(self.points.size(0), -1, 3) - self.points.transpose(2,
                                                                                              1).contiguous()) ** 2)
        loss_train_total.backward()
       # end=time.time()
       # print("loss backward time is ",end-start)
        self.log.update("loss_train_total", loss_train_total)
        self.optimizer.step()  # gradient update


        # VIZUALIZE
        """ 
        start = time.time()
        if self.iteration % 100 == 1 and self.opt.display:
            self.visualizer.show_pointclouds(points=self.points[0], title="train_input")
            self.visualizer.show_pointclouds(points=pointsReconstructed[0], title="train_input_reconstructed")
            if self.opt.dim_template == 3:
                self.visualizer.show_pointclouds(points=self.network.template[0].vertex, title=f"template0")
            if self.opt.patch_deformation and self.opt.dim_out_patch == 3:
                template = self.network.get_patch_deformation_template()
                end=time.time()
                print("VIZUALIZE time is ",end-start)
                self.network.train() #Add this or the training keeps going in eval mode!
                print("Network in TRAIN mode!")

                self.visualizer.show_pointclouds(points=template[0], title=f"template_deformed0")
        """
        self.print_iteration_stats(loss_train_total)

    def optim_reset(self, flag):
        if flag:
            # Currently I choose to reset the optimiser because it's to complicated to copy the branch optims
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.opt.lrate) # get new optimiser
            # optimizer.load_state_dict(self.optimizer.state_dict()) # copy state
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.opt.lrate)

    def train_epoch(self,epoch):
        self.log.reset()#初始化要输出的值 正确率 loss啥的
        self.network.train()#将网络调成训练模式
        self.learning_rate_scheduler()#作为回调函数的一员,LearningRateScheduler 可以按照epoch的次数自动调整学习率,
        start = time.time()
        iterator = self.dataloader_train.__iter__()
        self.reset_iteration()
        while True:
            try:
                # if self.iteration > 10:
                #     break
                points, idx, _, _ = iterator.next()
                points = points.transpose(2, 1).contiguous()
                points = points.cuda()
                self.points = points
                self.idx = idx
                self.increment_iteration()
            except:
                print(colored("end of train dataset", 'red'))
                break
            self.train_iteration(epoch)
        print("Ellapsed time : ", time.time() - start)

    def test_iteration(self):
        pointsReconstructed = self.Tran_points_test()

        loss_val_Deformation_ChamferL2 = torch.mean(
                (pointsReconstructed.view(self.points.size(0), -1, 3) - self.points.transpose(2, 1).contiguous()) ** 2)
        pointreloss=self.Reconstruction_loss_test()

        self.log.update("loss_val_Deformation_ChamferL2", loss_val_Deformation_ChamferL2)
        self.log.update("pointreloss", pointreloss)
        print(
            '\r' + colored('[%d: %d/%d]' % (self.epoch, self.iteration, self.len_dataset_test / (self.opt.batch_size)),
                           'red') +
            colored('loss_val_Deformation_ChamferL2:  %f' % loss_val_Deformation_ChamferL2.item(), 'yellow'),
            colored('pointreloss:  %f' % pointreloss.item(), 'yellow'),
            end='')

       # if self.iteration % 60 == 1 and self.opt.display:
            #self.visualizer.show_pointclouds(points=self.points[0], title="test_input")
            #self.visualizer.show_pointclouds(points=pointsReconstructed[0], title="test_input_reconstructed")

    def test_epoch(self):
        self.network.eval()
        iterator = self.dataloader_test.__iter__()
        self.reset_iteration()
        while True:
            self.increment_iteration()
            try:
                # if self.iteration > 10:
                #     break
                points, _, _, _ = iterator.next()
                points = points.transpose(2, 1).contiguous()
                points = points.cuda()
                self.points = points
               # print("Points test",points.size())
            except:
                print(colored("end of val dataset", 'red'))
                break
            self.test_iteration()

        self.log.end_epoch()
        if self.opt.display:
            #self.log.update_curves(self.visualizer.vis, self.opt.dir_name)
            self.network.save_template_png(self.opt.dir_name)
