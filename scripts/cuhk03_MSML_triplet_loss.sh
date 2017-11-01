now=$(date +"%Y%m%d_%H%M%S")

# export PATH=/mnt/lustre/share/anaconda2/bin:$PATH
export PATH=/mnt/lustre/sunhongbin/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-8.0-cudnn5.1/lib64/:$LD_LIBRARY_PATH
# if [ ! -d "log" ]; then
#   mkdir log
# fi

jobname=CUHK03-ResNet50-MSMLTripletLoss
num_gpus=2

log_dir=/mnt/lustre/sunhongbin/person_reid/cuhk03/examples/logs/MSML-triplet-loss/cuhk03-resnet50/tmp1101

srun -p TITANXP --job-name=$jobname --gres=gpu:$num_gpus  \
python -u examples/cuhk03_MSML_triplet_loss.py \
    -d cuhk03  \
    -a resnet50 \
    -b 128 \
    -j 2 \
    --lr 0.001 \
    --height 224 \
    --width 224 \
    --margin 0.3 \
    --epochs 400 \
    --combine-trainval \
    --logs-dir $log_dir \
