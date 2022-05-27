# fashion-print-inpainting
Detects prints on apparel and removes them using deep learning

## Download lama requirements
```
pip install -r lama/requirements.txt
```

## Download detr requirements
```
pip install -r requirements.txt
```
## Download the lama weights
Downlaod the lama weights in `lama` directory. `big-lama` folder is provided. Place it in `lama` directory

## Download the detr weights
Download the detr weights in `detr` directory. `checkpoints` folder is provided. Place it in `detr` directory
## Run the flask server
```
python app.py
```
## Send POST request
You might have to install `opencv-python` and `pillow`
```
pip install opencv-python
pip install pillow
```
If the requirements are already satisfied do:
```
python client.py data/test/imgs/test_img.png --threshold 0.7 --padding 20 --addr http://ec2-3-129-234-103.us-east-2.compute.amazonaws.com:5100
```

- threshold: Confidence threshold of bounding boxes

- padding: Increase in pixel size of bounding boxes

- addr: Address of server i.e http://ec2-3-129-234-103.us-east-2.compute.amazonaws.com:5100
