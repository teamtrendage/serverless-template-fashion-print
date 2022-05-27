
import argparse
import cv2
from matplotlib import pyplot as plt
import os
from PIL import Image
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json

def mask2bbox(np_mask, min_area=10):
    """
    returns
    bbox_list (list): list of tuple [(x,y,w,h)]"""
    contours, hierarchy = cv2.findContours(np_mask, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    
    bbox_list = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x,y,w,h = cv2.boundingRect(contour)
            bbox_list.append((x,y,w,h))
    return bbox_list

def convert_to_coco(imgs_dir, masks_dir, save_path="coco_dataset.json"):

    coco = Coco()
    coco.add_category(CocoCategory(id=0, name='print'))
    img_paths = [os.path.join(imgs_dir,f) for f in os.listdir(imgs_dir)]

    print("Starting the conversion...")
    for img_path in img_paths:
        img_path = os.path.abspath(img_path)
        file_name = os.path.basename(img_path)
        # mask_path = os.path.join(
        #     masks_dir,file_name.replace(".png","_boundingbox.png"))
        mask_path = os.path.join(masks_dir,file_name)
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h,w = img.shape

        bboxs = mask2bbox(img)

        coco_image = CocoImage(file_name=img_path,
                                height=h, width=w)
        for x_min,y_min,width,height in bboxs:
                
            coco_image.add_annotation(
                CocoAnnotation(
                bbox=[x_min, y_min, width, height],
                category_id=0,
                category_name='print'
                )
            )
        coco.add_image(coco_image)
    print("Finished conversion.")
    save_json(data=coco.json, save_path=save_path)
    print("Finished saving")



if __name__ == "__main__":
    convert_to_coco(
        "../data/train/imgs","../data/train/masks",
        save_path="../data/train/train_coco.json")
    convert_to_coco(
        "../data/val/imgs","../data/val/masks",
        save_path="../data/val/val_coco.json")
    convert_to_coco(
        "../data/test/imgs","../data/test/masks",
        save_path="../data/test/test_coco.json")
