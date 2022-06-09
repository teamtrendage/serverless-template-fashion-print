import argparse
import base64
import json

import cv2
import numpy as np
import requests

# addr = 'http://localhost:5000'
# addr = "http://ec2-3-129-234-103.us-east-2.compute.amazonaws.com:5100"
# test_url = addr + '/api/test'
def main(args):
    # prepare headers for http request
    addr = args.addr
    test_url = addr + '/'
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

    response = requests.post(test_url, json=json.dumps(model_inputs))

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
    parser.add_argument(
        '--addr',type=str,
        default='http://localhost:8000',
        help="Endpoint")
    args = parser.parse_args()
    main(args)