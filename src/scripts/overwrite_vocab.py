def compare_vocabulary(old_vocab_file, new_vocab_file):
    vocab_old = list()
    vocab_new = list()
    overlap = 0
    with open(old_vocab_file, 'r', encoding='utf8') as fd:
        old_vocab_data = fd.read().splitlines()
    with open(new_vocab_file, 'r', encoding='utf8') as fd:
        new_vocab_data = fd.read().splitlines()
    for data in old_vocab_data:
        word = data.split()[0]
        vocab_old.append(word)
    for data in new_vocab_data:
        word = data.split()[0]
        vocab_new.append(word)
    for word in vocab_old:
        if word in vocab_new:
            overlap += 1
    print(overlap, len(vocab_old), len(vocab_new))
    print(overlap / len(vocab_old) * 1.0)

def overwrite_vocab(old_vocab_file, new_vocab_file, over_vocab_file):
    vocab_old = dict()
    vocab_new = list()
    vocab_add = list()
    overlap = 0
    with open(old_vocab_file, 'r', encoding='utf8') as fd:
        old_vocab_data = fd.read().splitlines()
    with open(new_vocab_file, 'r', encoding='utf8') as fd:
        new_vocab_data = fd.read().splitlines()
    for data in old_vocab_data:
        word = data.split()[0]
        vocab_old[word] = data.split()[1]
    for data in new_vocab_data:
        word = data.split()[0]
        vocab_new.append((word, data.split()[1]))
        vocab_add.append(word)
    #print(vocab_new[:50])
    #vocab_new = sorted(vocab_new, key=lambda x: int(x[1]), reverse=True)
    #print(vocab_new[:50])
    not_overlap_vocab = dict()
    overwrite_vocab = dict()
    for new_word_tuple in vocab_new:
        new_word = new_word_tuple[0]
        if new_word not in vocab_old:
            not_overlap_vocab[new_word_tuple[0]] = new_word_tuple[1]
    # index_add = 0
    # with open(over_vocab_file, 'w', encoding='utf8') as fw:
    #     for word in not_overlap_vocab:
    #         fw.write(word[0] + ' ' + str(word[1]) + '\n')

    print(len(vocab_old))
    #print(len(vocab_add))
    print(len(not_overlap_vocab))
    for word in vocab_old:
        overwrite_vocab[word] = vocab_old[word]
    for word in not_overlap_vocab:
        overwrite_vocab[word] = not_overlap_vocab[word]
    print(len(overwrite_vocab))

    with open(over_vocab_file, 'w', encoding='utf8') as fw:
        for word in overwrite_vocab:
            fw.write(word + ' ' + overwrite_vocab[word] + '\n')

def check_vocab():
    check_c = 'c'
    vocab = dict()
    with open('/dockerdata/misc/model_dict.sum2.txt', 'r', encoding='utf8') as fd:
        vocab_data = fd.read().splitlines()    
    for data in vocab_data:
        word = data.split()[0]
        vocab[word] = data.split()[1]
    if check_c in vocab:
        print(check_c, vocab[check_c])
    else:
        print('false')


def main():
    old_vocab = ''
    new_vocab = ''
    overwrite_vocab(old_vocab, new_vocab, over_vocab_file='')

main()