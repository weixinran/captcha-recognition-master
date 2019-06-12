# -*- coding: UTF-8 -*-
import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
#from visdom import Visdom # pip install Visdom
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN
import one_hot_encoding
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_cuda = torch.cuda.is_available()

def main():
    cnn = CNN()
    if use_cuda:
        cnn = CNN().cuda()
    cnn.eval()
    cnn.load_state_dict(torch.load('model_splic_cos_90_0.01.pkl'))
    print("load cnn net.")

    predict_dataloader = my_dataset.get_label_data_loader()
    #vis = Visdom()
    correct = 0
    total = 0
    currentpath = os.getcwd()
    for i, (images,name) in enumerate(predict_dataloader):
        image = images
        vimage = Variable(image).cuda()
        predict_label = cnn(vimage).cpu()

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        #true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += images.size(0)
        # print(predict_label, true_label)
        # if (predict_label == true_label):
        #     correct += 1
        os.chdir(r"/home/captcha-recognition-master/dataset/label/")
        #now = str(int(time.time()))
        print(name[0],predict_label)
        os.rename(str(name[0]), (str(predict_label) + '.png'))  # 文件重新命名
        os.chdir(currentpath)  # 改回程序运行前的工作目录
        sys.stdin.flush()  # 刷新
        if (total % 200 == 0):
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    os.chdir(currentpath)  # 改回程序运行前的工作目录
    sys.stdin.flush()  # 刷新
        #vis.images(image, opts=dict(caption=c))

if __name__ == '__main__':
    main()


