# inference for M2M
export CUDA_VISIBLE_DEVICES=1

export OUTPUT_PATH='outputs/m2m100tiny/wmt6/'
export DATA_PATH='data/test/wmt6_m2m_bin/'
export CHECKPOINT_PATH='mPLM/418M_last_checkpoint.pt'
export lang_pairs=ps-en
export TASK=translation_multi_simple_epoch
export SRC=ps
export TGT=en

mkdir -p $OUTPUT_PATH

echo 'process the language pairs '$SRC'-'$TGT
fairseq-generate $DATA_PATH \
    --batch-size 64 \
    --path $CHECKPOINT_PATH \
    -s $SRC -t $TGT \
    --task $TASK \
    --remove-bpe 'sentencepiece' --beam 5 \
    --lang-pairs $lang_pairs \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test > $OUTPUT_PATH''$SRC'-'$TGT'.gen_out'
echo 'inference done!'