now=$(date +"%Y%m%d_%H%M%S")

# export PATH=/mnt/lustre/share/anaconda2/bin:$PATH
export PATH=/mnt/lustre/sunhongbin/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-8.0-cudnn5.1/lib64/:$LD_LIBRARY_PATH
# if [ ! -d "log" ]; then
#   mkdir log
# fi

jobname=Market1501-ResNet50-SDMLTripletLoss
num_gpus=2

log_dir=/mnt/lustre/sunhongbin/person_reid/cuhk03/examples/logs/SDML-triplet-loss/market1501-resnet50/tmp1101

srun -p TITANXP --job-name=$jobname --gres=gpu:$num_gpus  \
python -u examples/market1501_SDML_triplet_loss.py \
    -d market1501  \
    -a resnet50 \
    -b 128 \
    -j 2 \
    --margin 0.5 \
    --epochs 200 \
    --combine-trainval \
    --logs-dir $log_dir \
