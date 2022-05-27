import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
# dirname = os.path.dirname(dirname)
sys.path.append(dirname)
# sys.path.append("/home/nash/Desktop/projects/fashion/detr/utils")

from bbox_ops import rescale_bboxes
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
# colors for visualization

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def drawbbox(pil_image, coco_annotations, id2label={0: 'print'}):
    draw = ImageDraw.Draw(pil_image, "RGBA")

    for annotation in coco_annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x, y, w, h = tuple(box)
        draw.rectangle((x, y, x+w, y+h), outline='red', width=1)
        draw.text((x, y), id2label[class_idx], fill='green')

    plt.imshow(pil_image)
    plt.show()
    # draw.show()


def plot_results(pil_img, prob, boxes, id2label={0: 'print'}):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def visualize_predictions(image, outputs, threshold=0.9, padding=50):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    image2 = np.array(image)
    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    # plot results
    thickness = 2
    for p, (xmin,ymin,xmax,ymax) in zip(probas[keep].tolist(),bboxes_scaled.tolist()):
        cv2.rectangle(
            image2, (int(xmin-padding//2),int(ymin-padding//2)), 
            (int(xmax+padding//2),int(ymax+padding//2)),
            (255,0,0),thickness
        )
    return image2

def bbox2mask(bboxes, mask_h, mask_w, mask_padding=20):
    mask = np.zeros((mask_h, mask_w))
    thickness = -1 # Fill in rectange
    for _, xmin,ymin,xmax,ymax in bboxes:
        cv2.rectangle(
            mask, (int(xmin-mask_padding//2),int(ymin-mask_padding//2)), 
            (int(xmax+mask_padding//2),int(ymax+mask_padding//2)),
            (255,255,255),thickness
            )
    
    return mask

def process_predictions(image, outputs, threshold=0.9, mask_padding=20):
    """Return bounding boxes enclosing prints with probablities
    return:
    results : [(p,xmin,ymin,xmax,ymax)]"""
    # keep only predictions with confidence >= threshold
    w,h = image.size
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(),
                                    image.size)
    probs = probas[keep].tolist()
    bboxes_scaled = bboxes_scaled.tolist()
    results = [(p, xmin, ymin, xmax, ymax) 
                    for p,(xmin, ymin, xmax, ymax) in zip(probs,bboxes_scaled)]
    # print(results)
    mask = bbox2mask(results,h,w,mask_padding)

    return results, mask