#!/bin/bash
#SBATCH --job-name inference                  # 任务在 squeue 中显示任务名为 example
#SBATCH --output /home/AIChineseMedicine/huangky/reusePLM/logs/inference_uk_12B.%A_%a.out                 # 任务输出重定向至 [任务组id]_[组内序号].out
#SBATCH --gres gpu:a100:1                   # 任务申请使用一张 A100 GPU
#SBATCH --time 2-0:00:00                    # 任务最长运行 1 天 1 小时，超时任务将被杀死

# 任务 ID 通过 SLURM_ARRAY_TASK_ID 环境变量访问
# 上述行指定参数将传递给 sbatch 作为命令行参数
# 中间不可以有非 #SBATCH 开头的行

# 执行 sbatch 命令前先通过 conda activate [env_name] 进入环境
# 执行程序
echo ${SLURM_ARRAY_TASK_ID}
python -V
# inference for M2M
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PROJ_PATH='/home/AIChineseMedicine/huangky/reusePLM'
export DIRECTION='x2e'
export USER_PATH=$PROJ_PATH'/mkt_all'

export OUTPUT_PATH=$PROJ_PATH'/outputs/flores_uk_normalkd/'
# Mode: base / base_extend / ewc / adapter_old_version (new and old) / adapter_extend (new and old) 
export MODE='base'

export DATA_PATH=$PROJ_PATH'/data/test/flores_uk_m2m100tinykd_bin'
export NEW_DATA_PATH=$PROJ_PATH'/data/test/incre4_m2m100_bin'
export BASE_DATA_PATH=$PROJ_PATH'/data/test/incre4_m2m100_bin'

export CHECKPOINT_PATH=$PROJ_PATH'/mPLM/1.2B_last_checkpoint.pt'
#export CHECKPOINT_PATH=$PROJ_PATH'/storage/flores-x4bn2e-transbig-ffnReLULarge-krailmapfinal/checkpoints/checkpoint_best.pt'
#export CHECKPOINT_PATH=$PROJ_PATH'/storage/flores-en2bn-translarge-none-none/checkpoints/checkpoint_best.pt'
#export CHECKPOINT_PATH=$PROJ_PATH'/storage/wmt-x4ro2e-transbig-m2m100final-krailadaptive/checkpoints/checkpoint_best.pt'

export lang_pairs=uk-en
#export lang_pairs=ha-en,is-en,ja-en,pl-en,ps-en,ta-en
#export lang_pairs=en-ha,en-is,en-ja,en-pl,en-ps,en-ta,en-bn
#export lang_pairs=bn-en,de-en,ro-en,uk-en
#export lang_pairs=en-bn,en-de,en-ro,en-uk

#langs_list=(ha is ja pl ps ta)
#langs_list=(de uk bn ro)
langs_list=(uk)

old_langs_repo=(ha is ja pl ps ta)
new_langs_repo=(bn ro de zh uk)
export TASK=translation_multi_simple_epoch

mkdir -p $OUTPUT_PATH

for langs in ${langs_list[*]}; do
    if [ $DIRECTION == 'x2e' ]; then
        SRC=$langs
        TGT=en
    elif [ $DIRECTION == 'e2x' ]; then
        SRC=en
        TGT=$langs
    else
        echo 'translation direction is not provided! pls. check~ '
        break
    fi
    echo '----------------------------------------------------------'
    echo 'process the language pairs '$SRC'-'$TGT
    if [ $MODE == 'base' ]; then
        # mode: base model / reply / ft + const
        echo "Mode: Base ..."
        fairseq-generate $DATA_PATH \
            --skip-invalid-size-inputs-valid-test \
            --batch-size 16 \
            --path $CHECKPOINT_PATH \
            -s $SRC -t $TGT \
            --task $TASK \
            --remove-bpe 'sentencepiece' --beam 5 \
            --lang-pairs $lang_pairs \
            --decoder-langtok --encoder-langtok src \
            --gen-subset test > $OUTPUT_PATH''$SRC'-'$TGT'.gen_out'
        echo 'Base model inference done!'
    elif [ $MODE == 'base_extend' ]; then
        # mode: reply + extend
        # task: multi_adapter? old -> without vex version
        # arch: adapter
        # serial adapter: false, freeze all: false ; base_dict -> None
        echo "Mode: Vocabulary extend ..."
        fairseq-generate $DATA_PATH \
            --user-dir $USER_PATH \
            --batch-size 64 \
            --path $CHECKPOINT_PATH \
            -s $SRC -t $TGT \
            --task $TASK \
            --remove-bpe 'sentencepiece' --beam 5 \
            --lang-pairs $lang_pairs \
            --decoder-langtok --encoder-langtok src \
            --gen-subset test > $OUTPUT_PATH''$SRC'-'$TGT'.gen_out'
        echo 'Base model with vocabulary extension inference done!'
    elif [ $MODE == 'adapter_extend' ]; then
        #TASK=translation_multi_adapter_vex
        TASK=translation_multi_adapter
        echo "Mode: Adapter with task-"$TASK" ... it's the new version"
        if [[ ${old_langs_repo[@]/${langs}/} != ${old_langs_repo[@]} ]]; then
            echo 'belongs to old language pairs'
            USER_PATH=$PROJ_PATH'/mktkp'
            DATA_PATH=$BASE_DATA_PATH
        else
            echo 'belongs to new language pairs'
            USER_PATH=$USER_PATH
            DATA_PATH=$NEW_DATA_PATH
        fi
        fairseq-generate $DATA_PATH \
            --user-dir $USER_PATH \
            --batch-size 64 \
            --path $CHECKPOINT_PATH \
            -s $SRC -t $TGT \
            --task $TASK \
            --remove-bpe 'sentencepiece' --beam 5 \
            --lang-pairs $lang_pairs \
            --decoder-langtok --encoder-langtok src \
            --gen-subset test > $OUTPUT_PATH''$SRC'-'$TGT'.gen_out'
        echo 'Adapter model inference done!'
    elif [ $MODE == 'ewc' ]; then
        # Task: ewc
        # Arch: transformer
        echo "Mode: EWC ..."
        USER_PATH=$PROJ_PATH'/mkt'
        fairseq-generate $DATA_PATH \
            --user-dir $USER_PATH \
            --batch-size 64 \
            --path $CHECKPOINT_PATH \
            -s $SRC -t $TGT \
            --task $TASK \
            --remove-bpe 'sentencepiece' --beam 5 \
            --lang-pairs $lang_pairs \
            --decoder-langtok --encoder-langtok src \
            --gen-subset test > $OUTPUT_PATH''$SRC'-'$TGT'.gen_out'
        echo 'Ewc mode inference done!'
    else
        echo "Mode: None... exit 1."
    fi
done

# elif [ $MODE == 'adapter_old_version' ]; then
#     TASK=translation_multi_adapter
#     echo "Mode: Adapter with task-"$TASK" ..."
#     if [[ ${old_langs_repo[@]/${langs}/} != ${old_langs_repo[@]} ]]; then
#         echo 'belongs to old language pairs'
#         USER_PATH=$PROJ_PATH'/mktkp'
#     else
#         echo 'belongs to new language pairs'
#         USER_PATH=$PROJ_PATH'/mkt'
#     fi
#     fairseq-generate $DATA_PATH \
#         --user-dir $USER_PATH \
#         --batch-size 64 \
#         --path $CHECKPOINT_PATH \
#         -s $SRC -t $TGT \
#         --task $TASK \
#         --remove-bpe 'sentencepiece' --beam 5 \
#         --lang-pairs $lang_pairs \
#         --decoder-langtok --encoder-langtok src \
#         --gen-subset test > $OUTPUT_PATH''$SRC'-'$TGT'.gen_out'
#     echo 'Adapter model inference done!'