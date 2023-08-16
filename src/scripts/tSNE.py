import torch
import numpy as np
from openTSNE import TSNE
import tsneutil
from sklearn import datasets

def load_tensor_from_checkpoint(filename):
    check = torch.load(filename)
    tensor = torch.cat([check[_] for _ in check.keys()], 0).reshape(-1, 1024)
    return tensor



#lang_low = ['am', 'be', 'ha', 'ig', 'kk', 'kn', 'ky', 'mr', 'my', 'oc', 'or', 'ps', 'te', 'zu']     # 低资源 14种语言 flores
#ang_med = ['af', 'as', 'az', 'cy', 'ga', 'gl', 'gu', 'hi', 'ka', 'km', 'ku', 'ml', 'ne', 'pa', 'ta', 'tg', 'ur', 'uz', 'xh'] # 中资源 19种 flores
#lang_high = ['ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'es', 'et', 'fa', 'fi', 'fr', 'he', 'hr', 'hu', 'id', 'is', 'it', 'ja', 'ko', 'lt', 'mk', 'ms', 'mt', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv', 'th', 'tr', 'uk', 'vi', 'zh']   # 高资源 41种 flores
langs=['ha','is','ja','pl','ps','ta','ro','de','uk','bn']

if __name__ == '__main__':

    model_name = 'mclm' #方法名
    data_source = "flores"
    cover_english = 0
   
    lang = langs
    color_set = tsneutil.MOUSE_41X_COLORS
    
    if cover_english:
        lang.append("en")
    lang_num = len(lang)
    print('语言种类：', lang_num)

    jpg_name = '{}.{}'.format(lang_num, model_name)

    path = 'embedding/{}.encoder_representation.{}.pt'.format(model_name, lang[0])
    x = load_tensor_from_checkpoint(path).squeeze(1)
    sample_num = x.size()[0]
    y = torch.zeros([sample_num])
    for i, l in enumerate(lang[1:]):
        filename = 'embedding/{}.encoder_representation.{}.pt'.format(model_name, l)
        tx = load_tensor_from_checkpoint(filename).squeeze(1)
        x = torch.cat((x, tx), dim=0)
        sample_num = tx.size()[0]
        ty = torch.full([sample_num], i+1)
        y = torch.cat((y, ty), dim=0)

    perplexity = len(lang) - 1
    tsne = TSNE(
        perplexity=10,
        n_iter=1000,
        metric='euclidean',
        n_jobs=8,
        random_state=42
    )

    x = tsne.fit(x.cpu().numpy())
    tsneutil.plot(x, y.cpu().numpy(), colors=color_set, s=10, alpha=0.9, name=jpg_name, lang=lang)
    print('end')



# iris = datasets.load_iris()
# x, y = iris['data'], iris['target']
# tsne = TSNE(
#     perplexity=20,
#     n_iter=500,
#     metric='euclidean',
#     n_jobs=8,
#     random_state=42
# )
# embed = tsne.fit(x)
# tsneutil.plot(embed, y, colors=tsneutil.MOUSE_10X_COLORS, s=20, alpha=0.9)
# print('end')

