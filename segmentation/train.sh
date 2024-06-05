python3 main.py \
-mode train \
-support 1 \
-neighbor 50 \
-save /scratch/charvi/model.pkl \
-cuda 0 \
-epoch 100 \
-bs 4 \
-dataset /scratch/charvi/shapenetcore_partanno_segmentation_benchmark_v0 \
-point 1024 \
-record /scratch/charvi/record.log \
-interval 1000 




#######################################################
# For ShapeNet DataSet ......


# (Pdb) TOTAL_PARTS_NUM
# 50
# (Pdb) labels
# tensor([1, 1, 2,  ..., 0, 2, 1])
# (Pdb) points.shape
# torch.Size([1024, 3])
# (Pdb) lables.shape
# *** NameError: name 'lables' is not defined
# (Pdb) labels.shape
# torch.Size([1024])
# (Pdb) mask
# tensor([[1., 1., 1.,  ..., 0., 0., 0.],
#         [1., 1., 1.,  ..., 0., 0., 0.],
#         [1., 1., 1.,  ..., 0., 0., 0.],
#         ...,
#         [1., 1., 1.,  ..., 0., 0., 0.],
#         [1., 1., 1.,  ..., 0., 0., 0.],
#         [1., 1., 1.,  ..., 0., 0., 0.]])
# (Pdb) mask.shize
# *** AttributeError: 'Tensor' object has no attribute 'shize'
# (Pdb) mask.shape
# torch.Size([1024, 50])
# (Pdb) onehot.shape
# torch.Size([16])
# (Pdb) onehot
# tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# (Pdb) points.shape
# torch.Size([1024, 3])
# (Pdb) onehot.shape
# torch.Size([16])