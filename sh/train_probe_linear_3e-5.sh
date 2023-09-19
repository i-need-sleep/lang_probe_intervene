#!/bin/bash
#SBATCH --job-name=train_probe_linear_3e-5       # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=4           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=64GB                 # 最大内存
#SBATCH --time=12:00:00           # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yh2689@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=./logs/%x%A.out           # 正常输出写入的文件
#SBATCH --error=./logs/%x%A.err            # 报错信息写入的文件
#SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
#SBATCH -p gpu                   # 有GPU的partition
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

nvidia-smi
nvcc --version
cd /l/users/yichen.huang/lang_probe_intervene/code   # 切到程序目录

echo "START"               # 输出起始信息
source /apps/local/anaconda3/bin/activate tim          # 调用 virtual env
python -u train_probes.py \
    --name_root linear_3e-5 \
    --probe_type linear \
    --lr 3e-5
echo "FINISH"                       # 输出起始信息
