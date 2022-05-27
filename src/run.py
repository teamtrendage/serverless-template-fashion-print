# In this file, we define run_model
# It runs every time the server is called

import cv2


def run_model(model, img, threshold, padding):

    # do preprocessing
    # N/A for this example

    # run the model
    inpainted_img, mask, bbox_img = model(img, threshold=threshold, padding=padding)

    # do postprocessing

    inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR)
    # inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR)
    bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)

    return inpainted_img, bbox_img