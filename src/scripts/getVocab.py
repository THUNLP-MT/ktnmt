def get_vocabulary(data, vocab_file='/home/AIChineseMedicine/huangky/reusePLM/misc/base/model_dict.incre4.32k.txt'):
    vocab = dict()
    #langs_list = ['cs','de','es','et','fi','fr','hi','lv','lt','tr','ru','ro','zh','en','ha','ja','km','ps','pl','is','ta']
    langs_list = ['en','ro','de','bn','uk']
    #langs_list = ['en','zh','ja','ko','th','vi']
    i = 0
    print('the length of data: ', len(data))
    for line in data:
        i += 1
        word_list = line.strip().split()
        for word in word_list:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        if i % 100000 == 0:
            print('process the ', i ,' lines, the sum is: ', len(data))
    print('the length of vocab: ', len(vocab))
    with open(vocab_file, 'w', encoding='utf8') as fw:
        for v in vocab:
            fw.write(v + ' ' + str(vocab[v]) + '\n')
        for langs in langs_list:
            fw.write('__'+ langs +'__' + ' 1' + '\n')

def get_bi_vocabulary(data, vocab_file, langs=''):
    vocab = dict()
    i = 0
    print('the length of data: ', len(data))
    for line in data:
        i += 1
        word_list = line.strip().split()
        for word in word_list:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        if i % 100000 == 0:
            print('process the ', i ,' lines, the sum is: ', len(data))
    print('the length of vocab: ', len(vocab))
    with open(vocab_file, 'w', encoding='utf8') as fw:
        for v in vocab:
            fw.write(v + ' ' + str(vocab[v]) + '\n')
        fw.write('__en__' + ' 1' + '\n')
        fw.write('__'+ langs +'__' + ' 1' + '\n')

def revise_from_spm(vocab_file, new_vocab_file, langs):
    with open(vocab_file, 'r', encoding='utf8') as fd:
        vocab = fd.readlines()
    with open(new_vocab_file, 'w', encoding='utf8') as fw:

        for index, v in enumerate(vocab):
            v = v.strip().split()[0]
            if index > 2:
                fw.write(v + ' ' + str(len(vocab)-index) + '\n')
        fw.write('__en__' + ' 1' + '\n')
        fw.write('__'+ langs +'__' + ' 1' + '\n')

def main():
    file_name = '/home/AIChineseMedicine/huangky/reusePLM/data/train/incre4_32k_spm/all4.data'
    with open(file_name, 'r', encoding='utf8') as fd:
        data = fd.readlines()
    
    get_vocabulary(data, vocab_file='/home/AIChineseMedicine/huangky/reusePLM/misc/base/model_dict.incre4.32k.txt')

    # file_name_a = '/home/AIChineseMedicine/huangky/reusePLM/data/train/wmt_de_bi32k_spm/de-en.en'
    # file_name_b = '/home/AIChineseMedicine/huangky/reusePLM/data/train/wmt_de_bi32k_spm/de-en.de'
    # with open(file_name_a, 'r', encoding='utf8') as fd:
    #     data_a = fd.readlines() 
    # with open(file_name_b, 'r', encoding='utf8') as fd:
    #     data_b = fd.readlines()
    # data = data_a + data_b    
    # get_bi_vocabulary(data, vocab_file='/home/AIChineseMedicine/huangky/reusePLM/misc/base/model_dict.deen.32k.txt', langs='de')

    # vocab_file = '/home/AIChineseMedicine/huangky/reusePLM/misc/base/bnen.32k.vocab'
    # new_vocab_file = '/home/AIChineseMedicine/huangky/reusePLM/misc/base/model_dict.bnen.32k.txt'   
    # revise_from_spm(vocab_file, new_vocab_file, 'bn')

main()