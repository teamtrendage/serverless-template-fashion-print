import banana_dev as banana
import argparse
import base64
import json
import cv2
import numpy as np
import time

# python3 bananatest.py test_imgs/zalwomtops-2199-62912004bdcc642871a5a64c60f0b239.png --threshold 0.7 --padding 20

def main(args, api_key, model_key):
    threshold = args.threshold
    threshold = str(threshold).encode()
    threshold = base64.b64encode(threshold)
    padding = args.padding
    padding = str(padding).encode()
    padding = base64.b64encode(padding)
 
    path = args.img_path
    img = cv2.imread(path)
    _, img_encoded = cv2.imencode('.png', img)
    img_encoded = base64.b64encode(img_encoded).decode()

    # send http request with image and receive response
    model_inputs = {
        'data':  img_encoded,
        'threshold': threshold.decode('utf-8'),
        'padding': padding.decode('utf-8'),
    }    
    print("Waiting for Server Response....")
    response = banana.run(api_key, model_key, model_inputs)
    print("Done.")
    response = json.loads(response.text)

    w,h = response["size"]

    inpainted_img = response["inpainted_img"]
    inpainted_img = base64.b64decode(inpainted_img)
    inpainted_img = np.frombuffer(inpainted_img, np.uint8)
    inpainted_img = inpainted_img.reshape(h,w,3)

    cv2.imwrite("received_inpainted.png",inpainted_img)

    bbox_img = response["bbox_img"]
    bbox_img = base64.b64decode(bbox_img)
    bbox_img = np.frombuffer(bbox_img, np.uint8)
    bbox_img = bbox_img.reshape(h,w,3)
    cv2.imwrite("received_bbox.png",bbox_img)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help="path of the image")
    parser.add_argument('--threshold', type=float, help='confidence threshold',
                        default=0.7)
    parser.add_argument('--padding',type=int, 
                        help='Extra pixels to padd in bounding-box',
                        default=20)
    args = parser.parse_args()

    api_key = "777c010a-7781-4e14-be0e-dc84d9d3e2ee" # "YOUR_API_KEY"
    model_key = "822d4fae-4e80-4845-a219-c59794e1d34c" # "YOUR_MODEL_KEY"

    tic = time.time()
    main(args, api_key, model_key)
    toc = time.time()
    print(f"Time taken to run API {toc-tic:.4f}")
