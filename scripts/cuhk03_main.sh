now=$(date +"%Y%m%d_%H%M%S")

# export PATH=/mnt/lustre/share/anaconda2/bin:$PATH
export PATH=/mnt/lustre/sunhongbin/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-8.0-cudnn5.1/lib64/:$LD_LIBRARY_PATH
# if [ ! -d "log" ]; then
#   mkdir log
# fi

jobname=CUHK03-ResNet50-TripletLoss
num_gpus=2

log_dir=/mnt/lustre/sunhongbin/person_reid/cuhk03/examples/logs/triplet-loss/cuhk03-resnet50/tmp1101
# log_dir=/mnt/lustre/sunhongbin/person_reid/cuhk03/examples/logs/triplet-loss/cuhk03-resnet50/tmp1030

srun -p GTX1080 --job-name=$jobname --gres=gpu:$num_gpus  \
python -u examples/cuhk03_main.py \
    -d cuhk03  \
    -a resnet50 \
    -b 72 \
    --epochs 200 \
    --combine-trainval \
    --logs-dir $log_dir \
