# Script to run experiments for Efficacy prediction task 

exp_name="efficacy_task_planet"
python3 -u clf_train.py --hidden-size 800 --num-layers 2 --clf-base-data-path ../data/clf_data/PT --dataset-version 2 --encoder-path ../data/models/3u7di6ag/ckpt.pt --event-type all --freq-threshold 2 --trial-threshold 50 --log-dir ../trained-models --valid-every 1 --print-on-screen --batch-size 512 --gradient-accumulation-steps 2 --lr 0.001 --warm-up-steps 100 --num-epochs 100 --do-valid --max-grad-norm 1 --activation relu --weight-decay 0.001 --task-weight uniform --encoder-lr-factor 10 --dropout-prob 0 --norm-layer layernorm --encoder-layers-finetune embedding conv0 conv1 conv2 --label-column-name ae --gpus 0 0 0 0 0 --val-gpus 0 0 0 0 0 --split-strategy trial --nbr-concat --random-seed 1 \
    --default-task-name binary_pair_efficacy --binary-classification total-based \
    --wandb_project clinical_trials_efficacy --exp_name $exp_name \
  |& tee ../logs/train_${exp_name}.log.txt
