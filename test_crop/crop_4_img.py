import os
from PIL import Image
import time

in_dir = 'images_1000'
out_dir = 'images_crop'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

img_list = os.listdir(in_dir)
for img_file in img_list:
    print(img_file)
    img_name = os.path.splitext(os.path.basename(img_file))[0]
    img = Image.open(os.path.join(in_dir, img_file))
    img_size = img.size
    crop_width = round(img_size[0]/4)

    for ii in range(4):
        x1 = crop_width * ii
        y1 = 0
        x2 = crop_width * (ii + 1)
        y2 = img_size[1]
        img_c = img.crop([x1, y1, x2, y2])
        now = str(int(time.time()))
        img_crop_name = os.path.join(out_dir, img_name + '_' + now +'_'+ str(ii) + '.png')
        img_c.save(img_crop_name)
