# -*- coding: UTF-8 -*-
import os
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
    cnn.load_state_dict(torch.load('model_splic_cos_300_512_0.001.pkl'))
    print("load cnn net.")

    predict_dataloader = my_dataset.get_predict_data_loader()

    #vis = Visdom()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(predict_dataloader):
        image = images
        vimage = Variable(image).cuda()
        predict_label = cnn(vimage).cpu()

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        # c = '%s%s%s%s' % (c0, c1, c2, c3)
        # print(c)
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        # predict = '%s%s%s%s' % (c0, c1, c2, c3)
        # predict_label = one_hot_encoding.decode(predict.numpy()[0])
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        print(predict_label, true_label)
        if (predict_label == true_label):
            correct += 1
        if (total % 200 == 0):
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
        #vis.images(image, opts=dict(caption=c))

if __name__ == '__main__':
    main()


