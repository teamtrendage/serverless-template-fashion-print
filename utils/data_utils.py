import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

def _increase_mask_area(np_img):
    kernel = np.ones((10,10),np.uint8)
    cleaning_kernel = np.ones((20,20),np.uint8)
    # First clean image for noise masks
    opening = cv2.morphologyEx(np_img, cv2.MORPH_OPEN, kernel)
    dilated_img = cv2.dilate(opening, kernel, iterations=1)
    return dilated_img

def increase_mask_area(input_dir,output_dir="result"):
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="could be image or dir")
    parser.add_argument("--output", type=str, default="./output_mask")

    args = parser.parse_args()

    if os.path.isfile(args.input):
        imgs = [args.input]
    else:
        imgs = [os.path.join(args.input,f) for f in os.listdir(args.input)]
    
    count = 1
    for img in imgs:
        img_name = os.path.basename(img)
        img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        output_img = _increase_mask_area(img)
        cv2.imwrite(os.path.join(args.output,img_name),output_img)
        print(f"{count}: {img_name} is done")
        count += 1
        # fig,(ax1,ax2) = plt.subplots(1,2)
        # ax1.imshow(img)
        # ax1.set_title("original")
        # ax2.imshow(output_img)
        # ax2.set_title("modified")
        # plt.show()

