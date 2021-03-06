now=$(date +"%Y%m%d_%H%M%S")

# export PATH=/mnt/lustre/share/anaconda2/bin:$PATH
export PATH=/mnt/lustre/sunhongbin/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-8.0-cudnn5.1/lib64/:$LD_LIBRARY_PATH
# if [ ! -d "log" ]; then
#   mkdir log
# fi

jobname=Market1501-ResNet50-TripletLoss
num_gpus=4

log_dir=/mnt/lustre/sunhongbin/person_reid/cuhk03/examples/logs/triplet-loss/market1501-resnet50/tmp1017

srun -p Bigvideo --job-name=$jobname --gres=gpu:$num_gpus  \
python -u examples/market1501_main.py \
    -d market1501  \
    -a resnet50 \
    --combine-trainval \
    --logs-dir $log_dir \
