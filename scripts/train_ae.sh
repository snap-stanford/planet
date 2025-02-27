# Script to run experiments for Adverse Event prediction task 

# Temporal split for train/dev/test split
exp_name="AE_task_temporal_split_planet"
python -u clf_train.py --hidden-size 800 --num-layers 2 --clf-base-data-path ../data/clf_data/PT --dataset-version 2 --encoder-path ../data/models/3u7di6ag/ckpt.pt --event-type all --freq-threshold 2 --trial-threshold 50 --log-dir ../trained-models --valid-every 1 --print-on-screen --batch-size 1024 --gradient-accumulation-steps 1 --lr 0.001 --warm-up-steps 100 --num-epochs 100 --do-valid --max-grad-norm 1 --activation relu --weight-decay 0.001 --task-weight uniform --encoder-lr-factor 10 --dropout-prob 0 --norm-layer layernorm --encoder-layers-finetune embedding conv0 conv1 conv2 --label-column-name ae --gpus 0 0 0 0 0 0 --val-gpus 0 0 0 0 0 0 --split-strategy temporal-end-test --nbr-concat --default-task-name ae_clf_or --random-seed 1 --enrollment-filter 20 --AE_emb 1 \
    --wandb_project clinical_trials_AE --exp_name $exp_name \
  |& tee ../logs/train_${exp_name}.log.txt


# Drug-disease-trial split for train/dev/test split
exp_name="AE_task_drugDiseaseTrial_split_planet"
python -u clf_train.py --hidden-size 800 --num-layers 2 --clf-base-data-path ../data/clf_data/PT --dataset-version 2 --encoder-path ../data/models/3u7di6ag/ckpt.pt --event-type all --freq-threshold 2 --trial-threshold 50 --log-dir ../trained-models --valid-every 1 --print-on-screen --batch-size 1024 --gradient-accumulation-steps 1 --lr 0.001 --warm-up-steps 100 --num-epochs 100 --do-valid --max-grad-norm 1 --activation relu --weight-decay 0.001 --task-weight uniform --encoder-lr-factor 10 --dropout-prob 0 --norm-layer layernorm --encoder-layers-finetune embedding conv0 conv1 conv2 --label-column-name ae --gpus 0 0 0 0 0 0 --val-gpus 0 0 0 0 0 0 --split-strategy drug-disease-trial-test --nbr-concat --default-task-name ae_clf_or --random-seed 1 --enrollment-filter 20 --AE_emb 1 \
    --wandb_project clinical_trials_AE --exp_name $exp_name \
  |& tee ../logs/train_${exp_name}.log.txt


# Drug-trial split for train/dev/test split
exp_name="AE_task_drugTrial_split_planet"
python -u clf_train.py --hidden-size 800 --num-layers 2 --clf-base-data-path ../data/clf_data/PT --dataset-version 2 --encoder-path ../data/models/3u7di6ag/ckpt.pt --event-type all --freq-threshold 2 --trial-threshold 50 --log-dir ../trained-models --valid-every 1 --print-on-screen --batch-size 1024 --gradient-accumulation-steps 1 --lr 0.001 --warm-up-steps 100 --num-epochs 100 --do-valid --max-grad-norm 1 --activation relu --weight-decay 0.001 --task-weight uniform --encoder-lr-factor 10 --dropout-prob 0 --norm-layer layernorm --encoder-layers-finetune embedding conv0 conv1 conv2 --label-column-name ae --gpus 0 0 0 0 0 0 --val-gpus 0 0 0 0 0 0 --split-strategy drug-trial-test --nbr-concat --default-task-name ae_clf_or --random-seed 1 --enrollment-filter 20 --AE_emb 1 \
    --wandb_project clinical_trials_AE --exp_name $exp_name \
  |& tee ../logs/train_${exp_name}.log.txt


# Disease-trial split for train/dev/test split
exp_name="AE_task_diseaseTrial_split_planet"
python -u clf_train.py --hidden-size 800 --num-layers 2 --clf-base-data-path ../data/clf_data/PT --dataset-version 2 --encoder-path ../data/models/3u7di6ag/ckpt.pt --event-type all --freq-threshold 2 --trial-threshold 50 --log-dir ../trained-models --valid-every 1 --print-on-screen --batch-size 1024 --gradient-accumulation-steps 1 --lr 0.001 --warm-up-steps 100 --num-epochs 100 --do-valid --max-grad-norm 1 --activation relu --weight-decay 0.001 --task-weight uniform --encoder-lr-factor 10 --dropout-prob 0 --norm-layer layernorm --encoder-layers-finetune embedding conv0 conv1 conv2 --label-column-name ae --gpus 0 0 0 0 0 0 --val-gpus 0 0 0 0 0 0 --split-strategy disease-trial-test --nbr-concat --default-task-name ae_clf_or --random-seed 1 --enrollment-filter 20 --AE_emb 1 \
    --wandb_project clinical_trials_AE --exp_name $exp_name \
  |& tee ../logs/train_${exp_name}.log.txt
