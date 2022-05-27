import base64
import time

import cv2
import jsonpickle
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from PIL import Image

from models.fashion_inpaint import FashionInpaint

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    threshold = r.headers['threshold']
    threshold = base64.b64decode(threshold)
    threshold = float(threshold)
    padding = r.headers['padding']
    padding = base64.b64decode(padding)
    padding = int(padding)
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(img)
    w,h = img.size
    inpainted_img,mask,bbox_img = fashion_model(img,threshold=threshold,padding=padding)
    
    enc_inpaint = base64.b64encode(inpainted_img).decode('utf-8')
    enc_bbox_img = base64.b64encode(bbox_img).decode('utf-8')
    # return Response(response=response_pickled, status=200, mimetype="application/json")
    return jsonify({
                'msg': 'success', 
                'size': [w, h], 
                'format': "PNG",
                'inpainted_img': enc_inpaint,
                'bbox_img': enc_bbox_img
           })

# start flask app
if __name__=="__main__":
    print("Loading model...")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    fashion_model = FashionInpaint(device=device)
    print("Model loaded.")
    app.run(host="0.0.0.0", port=5000)