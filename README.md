# KTNMT

## Introduction
Source code for the ACL 2023 main conference long paper "Knowledge Transfer in Incremental Learning for Multilingual Neural Machine Translation" (Outstanding Paper Award)

In this work, we propose a knowledge transfer method that can efficiently adapt original MNMT models to diverse incremental language pairs.

---
## Get Started
(Core) Data Preprocessing.

Standard BPE Procedure: following https://github.com/google/sentencepiece with 64k merged BPE tokens.

After obtaining the original vocabulary and the incremental vocabulary, you must get an incremental vocabulary of external models.

Model Training:

This system has been tested in the following environment.

Python version == 3.7

Pytorch version == 1.8.0

Fairseq version == 0.12.0 (pip install fairseq)

Note that it only influences the training procedure of the original and incremental model. You can choose your favorite deep learning library for model training.

---
## Incremental Learning
We build the incremental learning procedure for Multilingual Neural Machine Translation as follows:

1. Get original multilingual translation models (or train a multilingual translation model by yourself).

2. Preprocessing incremental data
- Data Clean (optional, if needed)
- Get Vocabulary (optional, if needed, follow standard BPE procedure)

3. Incremental Training.
    We provide all runing scripts in the folder ''src''. 
    Here is an example:
    ```bash
    echo "Task format: "
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export SEED=222
    export CHECKPOINT_PATH='' # save new model
    export CL_CHECKPOINT_PATH='' # old model
    export DATA_PATH='' # incremental data bin
    export USER_PATH='mkt/' # user dir
    export ARCH=transformer_adapter
    export TASK=translation_multi_adapter
    export ADAPTER_MODULE=serial # serial / parallel / parallel_gate / parallel_norm / parallel_dp
    export KRAIL_MODULE='' # krail / krail_ffn / krail_map / krail_map_last / krail_map_avg / krail_adaptive / krail_adaptive_final
    export LANG_DICT='misc/cl_adapter/lang_dicts_uk.txt' # new lang dict
    export KRAIL_MODEL='mPLM/418M_last_checkpoint.pt' # external model path
    export NEW_DICT_PATH='misc/cl_adapter/model_dict.m2m100.vex.txt' # combination model dict, we can get this vocab by scripts
    export KRAIL_DICT_PATH='misc/m2m100/model_dict.128k.txt' # external model dict
    export lang_pairs=en-uk
    export CLIP_NORM=0.0
    export OPTIMIZER=adam
    export ADAM_EPS=1e-9
    export LR=5e-4
    export LR_SCHEDULER=inverse_sqrt
    export WARMUP=4000
    export DROPOUT=0.2
    export ATT_DROPOUT=0.2
    export WEIGHT_DECAY=0.0001
    export CRITERION=label_smoothed_cross_entropy
    export LABEL_SMOOTHING=0.1
    export MAX_TOKENS=1024
    export UPDATE_FREQ=8
    export SAVE_INTERVAL_UPDATES=2500
    export KEEP_INTERVAL_UPDATES=1
    export MAX_UPDATE=500000

    fairseq-train $DATA_PATH \
    --user-dir $USER_PATH \
    --share-all-embeddings \
    --encoder-normalize-before --decoder-normalize-before \
    --encoder-embed-dim 1024 --encoder-ffn-embed-dim 8192 --encoder-attention-heads 16 \
    --decoder-embed-dim 1024 --decoder-ffn-embed-dim 8192 --decoder-attention-heads 16 \
    --encoder-layers 24 --decoder-layers 24 \
    --left-pad-source False --left-pad-target False \
    --arch $ARCH \
    --task $TASK \
    --sampling-method temperature \
    --sampling-temperature 5 \
    --lang-tok-style multilingual \
    --lang-dict $LANG_DICT \
    --lang-pairs $lang_pairs \
    --encoder-langtok src \
    --decoder-langtok \
    --clip-norm $CLIP_NORM \
    --optimizer $OPTIMIZER \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps $ADAM_EPS \
    --lr $LR \
    --lr-scheduler $LR_SCHEDULER \
    --warmup-updates $WARMUP \
    --dropout $DROPOUT \
    --attention-dropout $ATT_DROPOUT \
    --weight-decay $WEIGHT_DECAY \
    --criterion $CRITERION \
    --label-smoothing $LABEL_SMOOTHING \
    --max-tokens $MAX_TOKENS \
    --save-dir $CHECKPOINT_PATH/checkpoints/ \
    --update-freq $UPDATE_FREQ \
    --save-interval-updates $SAVE_INTERVAL_UPDATES \
    --keep-interval-updates $KEEP_INTERVAL_UPDATES \
    --max-update $MAX_UPDATE \
    --no-epoch-checkpoints \
    --seed $SEED --log-format simple --log-interval 300 \
    --patience 10 \
    --tensorboard-logdir $CHECKPOINT_PATH/logs/ \
    --no-progress-bar \
    --ddp-backend no_c10d \
    --finetune-from-model $CL_CHECKPOINT_PATH \
    --freeze-all \
    --adapter-inner-dims 4096 \
    --adapter-module $ADAPTER_MODULE \
    --krail-module $KRAIL_MODULE \
    --vocab-adapter \
    --base-embed-dim 32211 \
    --krail-model $KRAIL_MODEL \
    --new-dict-path $NEW_DICT_PATH \
    --krail-dict-path $KRAIL_DICT_PATH
    ```
---
## Inference & Evaluation
Please refer to src/inference.sh & run_sh/evaluate.sh

---
## Citation
```
@inproceedings{huang-etal-2023-knowledge,
    title = "Knowledge Transfer in Incremental Learning for Multilingual Neural Machine Translation",
    author = "Huang, Kaiyu  and
      Li, Peng  and
      Ma, Jin  and
      Yao, Ting  and
      Liu, Yang",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.852",
    doi = "10.18653/v1/2023.acl-long.852",
    pages = "15286--15304",
}
```