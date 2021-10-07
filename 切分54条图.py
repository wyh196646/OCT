#python install pillow
from os import path
from PIL import Image
from pathlib import *
import pandas as pd
 
def cut_and_save_image(origin_image_path,object_image_path,count=54):
    image=Image.open(origin_image_path)
    width, height = image.size
    item_width = int(width / count) 
    item_height = height
    box_list = []
    # (left, upper, right, lower)
    for i in range(1,count-1):
            box = ((i-1)*item_width,0,(i+2)*item_width,item_height)#parameter si left top right bottom 
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    index = 2
    for image in image_list:
        image.save(str(object_image_path)+'_'+str(index) + '.jpg')
        index += 1
    return image_list


data=pd.read_csv('data.csv')
original_image_path=Path('/home/octusr3/project/data_fast/380')
image_paths=data['image_path']
object_image_path=Path('/home/octusr3/project/data_fast/54')

if __name__=='__main__':
    for image_path in image_paths:
        Path(object_image_path/image_path).parent.mkdir(parents=True,exist_ok=True)
        cut_and_save_image(str(original_image_path/image_path),Path(object_image_path/image_path).with_name(Path(image_path).stem))