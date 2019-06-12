# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
import captcha_setting
from captcha_cnn_model import CNN
import torch.backends.cudnn as cudnn
import one_hot_encoding

# Hyper Parameters
num_epochs = 20
batch_size = 100
learning_rate = 0.0005
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_cuda = torch.cuda.is_available()

def main():
    cnn = CNN()

    #resume
    resume = './model_cos_90_0.01.pkl'
    checkpoint = torch.load(resume)
    cnn.load_state_dict(checkpoint)

    if use_cuda:
        cnn = CNN().cuda()
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    best_acc = 0
    # Train the Model
    train_dataloader = my_dataset.get_finetune_data_loader()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_dataloader):
            #lr_schedule
            lr = cosine_anneal_schedule(epoch)

            for param_group in optimizer.param_groups:
                #print(param_group['lr'])
                param_group['lr'] = lr

            images = Variable(images).cuda()
            labels = Variable(labels.float()).cuda()
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predict_labels = predict_labels.cpu()
            labels = labels.cpu()

            c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_labels[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_labels[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_labels[0,
                                                        2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_labels[0,
                                                        3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

            # c = '%s%s%s%s' % (c0, c1, c2, c3)
            # print(c)
            predict_labels = '%s%s%s%s' % (c0, c1, c2, c3)
            # predict = '%s%s%s%s' % (c0, c1, c2, c3)
            # predict_label = one_hot_encoding.decode(predict.numpy()[0])
            true_label = one_hot_encoding.decode(labels.numpy()[0])
            total += labels.size(0)
            print(predict_labels, true_label)
            if (predict_labels == true_label):
                correct += 1
            acc = 100 * correct / total
            # if (total % 200 == 0):
            #     print('Test Accuracy of the model on the %d train images: %f %%' % (total, 100 * correct / total))
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item(),"Accuracy:", acc)
            if acc >= best_acc:
                torch.save(cnn.state_dict(), "./model_cos_90_0.01_finetune_2.pkl")   #current is model.pkl
                print("save model")
            #print("epoch:", epoch, "loss:", loss.item(),"Accuracy:", acc)
    #torch.save(cnn.state_dict(), "./model_cos_90_0.01_finetune.pkl")   #current is model.pkl
    print("save last model")

def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (num_epochs))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (num_epochs)
    cos_out = np.cos(cos_inner) + 1
    return float(learning_rate / 2 * cos_out)

if __name__ == '__main__':
    main()


