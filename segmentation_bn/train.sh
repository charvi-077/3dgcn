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
-interval 1000 \