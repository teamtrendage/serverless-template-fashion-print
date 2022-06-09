import base64
import time

import cv2
import jsonpickle
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from PIL import Image
import json
from warmup import load_model
from run import run_model

# do the warmup step globally, to have a reuseable model instance
print('Loading Model....')
device = 'cuda'
model = load_model(device)
print('Model Loaded!')




# Initialize the Flask application
app = Flask(__name__)



@app.route('/healthcheck', methods=["GET"])
def healthcheck():
    r = request
    return jsonify({"state": "healthy"})


# route http posts to this method
@app.route('/', methods=['POST'])
def inference():
    r = request
    # Reading request json 
    req = json.loads(r.json)
    
    # extracting and decoding threshold, padding 
    threshold = req['threshold']
    threshold = base64.b64decode(threshold)
    threshold = float(threshold)
    padding = req['padding']
    padding = base64.b64decode(padding)
    padding = int(padding)
    nparr = np.fromstring(req['data'], np.uint8)

    # extracting and decoding image    
    nparr = base64.b64decode(nparr)
    nparr = np.frombuffer(nparr, dtype=np.uint8)
    
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(img)
    w,h = img.size
    
    # Predicting    
    inpainted_img, bbox_img = run_model(model, img, threshold, padding)
    
    enc_inpaint = base64.b64encode(inpainted_img).decode('utf-8')
    enc_bbox_img = base64.b64encode(bbox_img).decode('utf-8')
    return jsonify({
                'msg': 'success', 
                'size': [w, h], 
                'format': "PNG",
                'inpainted_img': enc_inpaint,
                'bbox_img': enc_bbox_img
           })

# start flask app
if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)