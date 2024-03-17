#!/bin/bash
#SBATCH --partition=a6000
#SBATCH -J region_withoutm2m_new_proposal_30modes
#SBATCH --gres=gpu:8
#SBATCH --nodelist=3dimage-21

python /space/renjx/qcnet_region/train_qcnet.py --root /datasets/Argoverse2/motion/ --train_batch_size 2 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 --val_batch_size 2 --test_batch_size 2 --train_raw_dir /datasets/Argoverse2/motion/train/ --val_raw_dir /datasets/Argoverse2/motion/val/ --test_raw_dir /datasets/Argoverse2/motion/test/ --train_processed_dir /datasets/qcnet/train --val_processed_dir /datasets/qcnet/val --test_processed_dir /datasets/qcnet/test