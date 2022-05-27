import base64
import time

import cv2
import jsonpickle
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from PIL import Image

from warmup import load_model
from run import run_model

# do the warmup step globally, to have a reuseable model instance
print('Loading Model....')
device = 'cpu'
model = load_model(device)
print('Model Loaded!')




# Initialize the Flask application
app = Flask(__name__)



@app.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    return response.json({"state": "healthy"})


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
    inpainted_img, bbox_img = run_model(model, img, threshold, padding)
    
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
    app.run(host="0.0.0.0", port=8000)