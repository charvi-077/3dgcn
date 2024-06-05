import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import gcn3d

class GCN3D(nn.Module):
    def __init__(self, class_num, support_num, neighbor_num):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 128, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num= support_num)

        dim_fuse = sum([128, 128, 256, 256, 512, 512, 16])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, class_num, 1),
        )

    def forward(self, 
                vertices: "tensor (bs, vetice_num, 3)",  # torch.Size([4, 1024, 3])
                onehot: "tensor (bs, cat_num)"):   # torch.Size([4, 16])
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size() # 4, 1024
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num) # torch.Size([4, 1024, 50]) self.neighbour number = 50
        # breakpoint() 
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace= True) # torch.Size([4, 1024, 128])
        fm_1 = F.relu(self.conv_1(neighbor_index, vertices, fm_0), inplace= True) # torch.Size([4, 1024, 128])
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1) # torch.Size([4, 256, 3]) # torch.Size([4, 256, 128])
        # breakpoint() 
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)  # torch.Size([4, 256, 50])
        # breakpoint() 

        fm_2 = F.relu(self.conv_2(neighbor_index, v_pool_1, fm_pool_1), inplace= True) # torch.Size([4, 1024, 256])
        fm_3 = F.relu(self.conv_3(neighbor_index, v_pool_1, fm_2), inplace= True) # torch.Size([4, 1024, 256])
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3) # torch.Size([4, 64, 3]) # torch.Size([4, 64, 256])
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num) # torch.Size([4, 64, 50])
        # breakpoint() 

        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2) # torch.Size([4, 64, 512])
        f_global = fm_4.max(1)[0] #(bs, f) # torch.Size([4, 512])
        # breakpoint() 

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1) # torch.Size([4, 1024, 1])
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2) # torch.Size([4, 1024, 1])
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2) # torch.Size([4, 1024, 256])
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2) # torch.Size([4, 1024, 256])
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2) # torch.Size([4, 1024, 512])
        f_global = f_global.unsqueeze(1).repeat(1, vertice_num, 1) # torch.Size([4, 1024, 512])
        onehot = onehot.unsqueeze(1).repeat(1, vertice_num, 1) #(bs, vertice_num, cat_one_hot) # torch.Size([4, 1024, 16])
        fm_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, f_global, onehot], dim= 2)  # torch.Size([4, 1024, 1808])
        # breakpoint()
        
        conv1d_input = fm_fuse.permute(0, 2, 1) #(bs, fuse_ch, vertice_num) # torch.Size([4, 1808, 1024])
        conv1d_out = self.conv1d_block(conv1d_input)  # torch.Size([4, 50, 1024])
        pred = conv1d_out.permute(0, 2, 1) #(bs, vertice_num, ch) # torch.Size([4, 1024, 50])
        # breakpoint()
        
        return pred

def test():
    from dataset_shapenet import test_model
    dataset = "../../shapenetcore_partanno_segmentation_benchmark_v0"
    model = GCN3D(class_num= 50, support_num= 1, neighbor_num= 50)
    test_model(model, dataset, cuda= "0", bs= 2, point_num= 2048)

if __name__ == "__main__":
    test()
    
    
########################## EXample shapes of my forward function ##############3
# (Pdb) vertices.shape
# torch.Size([4, 1024, 3])
# (Pdb) onehot.shape
# torch.Size([4, 16])
# (Pdb) bs
# 4
# (Pdb) vertice_num
# 1024
# (Pdb) self.neighbor_num
# 50
# (Pdb) neighbor_index.shape
# torch.Size([4, 1024, 50])

# (Pdb) fm_0.shape
# torch.Size([4, 1024, 128])
# (Pdb) fm_1.shape
# torch.Size([4, 1024, 128])
# (Pdb) v_pool_1.shape
# torch.Size([4, 256, 3])
# (Pdb) fm_pool_1.shape
# torch.Size([4, 256, 128])

# (Pdb) neighbor_index.shape
# torch.Size([4, 256, 50])

# (Pdb) fm_2.shape
# torch.Size([4, 1024, 256])
# (Pdb) fm_3.shape
# torch.Size([4, 1024, 256])
# (Pdb) v_pool_2.shape
# torch.Size([4, 64, 3])
# (Pdb) fm_pool_2.shape
# torch.Size([4, 64, 256])
# (Pdb) neighbor_index.shape
# torch.Size([4, 64, 50])

# (Pdb) fm_4.shape
# torch.Size([4, 64, 512])
# (Pdb) f_global.shape
# torch.Size([4, 512])

# (Pdb) nearest_pool_1.shape
# torch.Size([4, 1024, 1])
# (Pdb) nearest_pool_2.shape
# torch.Size([4, 1024, 1])
# (Pdb) fm_2.shape
# torch.Size([4, 1024, 256])
# (Pdb) fm_3.shape
# torch.Size([4, 1024, 256])
# (Pdb) fm_4.shape
# torch.Size([4, 1024, 512])
# (Pdb) f_global.shape
# torch.Size([4, 1024, 512])
# (Pdb) one_hot.shape
# *** NameError: name 'one_hot' is not defined
# (Pdb) onehot.shape
# torch.Size([4, 1024, 16])
# (Pdb) fm_fuse.shape
# torch.Size([4, 1024, 1808])

# (Pdb) conv1d_input.shape
# torch.Size([4, 1808, 1024])
# (Pdb) conv1d_out.shape
# torch.Size([4, 50, 1024])
# (Pdb) pred.shape
# torch.Size([4, 1024, 50])


