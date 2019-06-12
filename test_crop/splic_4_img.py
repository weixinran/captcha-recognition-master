import os
import random
import time
# import numpy as np
from PIL import Image

in_dir = 'images_crop'
out_dir = 'images_splic_1000'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

img_list = os.listdir(in_dir)
count = 1000000
for i in range(count):
    splic = Image.new('RGB', (75, 24))
    img_label = ''
    for j in range(4):
        index = random.randint(0, len(img_list) - 1)
        image = Image.open(os.path.join(in_dir, img_list[index]))

        if j == 0:
            left = 0
            right = image.size[0]

        img_name = os.path.splitext(os.path.basename(img_list[index]))[0]
        img_label += img_name.split('_')[0][int(img_name.split('_')[2])]

        splic.paste(image, (left, 0, right, image.size[1]))  # 将image复制到target的指定位置中
        left += image.size[0]  # left是左上角的横坐标，依次递增
        right += image.size[0]  # right是右下的横坐标，依次递增
        quality_value = 100  # quality来指定生成图片的质量，范围是0～100

        now = str(int(time.time()))
        img_splic_name = os.path.join(out_dir, img_label + '_' + now + '.png')
    splic.save(img_splic_name, quality=quality_value)
    print(i, img_splic_name)
    # img = np.array(Image.open(os.path.join(in_dir, img_list[index])))
    # print(img.shape)
    # array_splic = np.concatenate((array_splic,img),axis=0)
    # img_splic = Image.fromarray(array_splic)
    # img_splic.save("your_file.jpeg")

# #crop
# for i,img_file in enummerate(img_list):
#     print(img_file)
#     img_name = os.path.splitext(os.path.basename(img_file))[0]
#     img = Image.open(os.path.join(in_dir, img_file))
#     img_size = img.size
#     crop_width = round(img_size[0]/4)

#     for ii in range(4):
#         x1 = crop_width * ii
#         y1 = 0
#         x2 = crop_width * (ii + 1)
#         y2 = img_size[1]
#         img_c = img.crop([x1, y1, x2, y2])
#         img_crop_name = os.path.join(out_dir, img_name + '_' + str(ii) + '.png')
#         img_c.save(img_crop_name)
