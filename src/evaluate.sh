#!/bin/bash
export DIRECTION='x2e'
export GEN_PATH='outputs/flores_uk_m2m100largetiny_extend/'

export REF_PATH='data/test/'
export OLD_REF_PATH='/data/test/'
export NEW_REF_PATH='/data/test/'
langs_list=(ha is ja pl ps ta)

old_langs_repo=(ha is ja pl ps ta)
new_langs_repo=(bn ro de zh uk)

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
    if [[ ${old_langs_repo[@]/${langs}/} != ${old_langs_repo[@]} ]]; then
        echo 'belongs to old language pairs'
        REF_PATH=$OLD_REF_PATH
    else
        echo 'belongs to new language pairs'
        REF_PATH=$NEW_REF_PATH
    fi
    if [ $TGT == 'zh' ]; then
        cat $GEN_PATH''$SRC'-'$TGT'.gen_out' | grep -P "^H" | sort -V | cut -f 3- > $GEN_PATH''$SRC'-'$TGT'.hyp'
        sacrebleu -w 2 -tok 'zh' $REF_PATH''$SRC'-'$TGT'.'$TGT < $GEN_PATH''$SRC'-'$TGT'.hyp' > $GEN_PATH''$SRC'-'$TGT'.eval'
    elif [ $TGT == 'ja' ]; then
        cat $GEN_PATH''$SRC'-'$TGT'.gen_out' | grep -P "^H" | sort -V | cut -f 3- > $GEN_PATH''$SRC'-'$TGT'.hyp'
        sacrebleu -w 2 -tok 'ja-mecab' $REF_PATH''$SRC'-'$TGT'.'$TGT < $GEN_PATH''$SRC'-'$TGT'.hyp' > $GEN_PATH''$SRC'-'$TGT'.eval'
    else
        cat $GEN_PATH''$SRC'-'$TGT'.gen_out' | grep -P "^H" | sort -V | cut -f 3- > $GEN_PATH''$SRC'-'$TGT'.hyp'
        sacrebleu -w 2 -tok '13a' $REF_PATH''$SRC'-'$TGT'.'$TGT < $GEN_PATH''$SRC'-'$TGT'.hyp' > $GEN_PATH''$SRC'-'$TGT'.eval'
    fi
    echo 'Translation directions (X->Y): '$SRC'-'$TGT
    cat $GEN_PATH''$SRC'-'$TGT'.eval' | grep "\"score\"" | cut -d : -f 2
    
done