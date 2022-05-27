import argparse
import base64
import os
import sys
import time

import cv2
import jsonpickle
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image

filepath = os.path.abspath(__file__)
dirpath = os.path.dirname(os.path.dirname(filepath))
sys.path.append(dirpath)
from transformers import DetrFeatureExtractor

from detr.models.detr import Detr
from detr.utils.visualization import process_predictions, visualize_predictions
from lama.saicinpainting.training.trainers import load_checkpoint


class FashionInpaint:

    def __init__(self, device='cuda'):
        # Models
        self.parent_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.parent_dir) 
        self.device = device
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50")
        detector_path = os.path.join(
            self.parent_dir,'detr','checkpoints','lightning_logs','version_1',
            'checkpoints','epoch=9-step=15510.ckpt')

        self.detector = Detr.load_from_checkpoint(detector_path,
        lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
        train_loader=None,
        val_loader=None
        )  
        self.detector.to(device)
        self.detector.eval()

        lama_path = os.path.join(self.parent_dir,'lama','big-lama')
        train_config_path = os.path.join(lama_path,"config.yaml")
        train_config = OmegaConf.load(train_config_path)
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        lama_checkpoint = os.path.join(lama_path,'models','best.ckpt')
        self.lama_model = load_checkpoint(
            train_config, lama_checkpoint, strict=False, map_location='cpu')
        self.lama_model.freeze()
        self.lama_model.to(device)
    
    def __call__(self,img, threshold=0.7, padding=20):

        results, mask, bbox_img = self.run_object_detector(
            img,threshold,padding)
        img = np.array(img)
        # mask = mask.astype(np.float32)

        resize_multiple = 8
        orig_h,orig_w = mask.shape
        new_h = int(resize_multiple * np.ceil(orig_h / resize_multiple))
        new_w = int(resize_multiple * np.ceil(orig_w / resize_multiple))
        img = cv2.resize(img,(new_w,new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask,(new_w,new_h), interpolation=cv2.INTER_AREA)

        inpainted_img = self.run_lama(img, mask)
        inpainted_img = cv2.resize(inpainted_img,(orig_w,orig_h),
                                    interpolation=cv2.INTER_AREA)
        return inpainted_img, mask, bbox_img

    def run_object_detector(self, pil_img, threshold=0.7, padding=20):

        encoding = self.feature_extractor(images=pil_img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
            
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        # forward pass to get class logits and bounding boxes
        outputs = self.detector(pixel_values=pixel_values, pixel_mask=None)
        results, np_mask = process_predictions(pil_img,outputs,
                                            threshold=threshold,
                                            mask_padding=padding)
        bbox_img = visualize_predictions(pil_img,outputs,threshold=threshold,
                                        padding=padding)
        # bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)

        return results, np_mask, bbox_img
    
    def run_lama(self, np_img, np_mask):

        img = np_img.astype(np.float32)
        img = img/255.0
        img = torch.from_numpy(img)
        img = img.permute(2,0,1).unsqueeze(0) #HWC to BCHW
        img = img.type(torch.FloatTensor).to(self.device)
        
        mask = np_mask.astype(np.float32)
        mask = mask/255.0
        mask = torch.from_numpy(mask).unsqueeze(-1)
        mask = mask.permute(2,0,1).unsqueeze(0) #HWC to BCHW
        mask = mask.type(torch.FloatTensor).to(self.device)

        batch = {'image':img, 'mask':mask}
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = self.lama_model(batch)
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir")
    parser.add_argument('--result_dir')

    args = parser.parse_args()

    img_dir = args.img_dir
    result_dir = args.result_dir

    imgs = [os.path.join(img_dir,f) for f in os.listdir(img_dir)]
    tic = time.time()
    fashion_model = FashionInpaint(device='cpu')
    toc1 = time.time()
    print(f"Time loading model {toc1-tic}")
    for img_path in imgs:
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)
        inpainted_img, mask, bbox_img = fashion_model(img,padding=20)
        inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR)
        # inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR)
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_dir,img_name.replace(".png","_p.png")),
        inpainted_img)
        cv2.imwrite(os.path.join(result_dir,img_name.replace(".png","_b.png")),
        bbox_img)
        cv2.imwrite(os.path.join(result_dir,img_name.replace(".png","_b.png")),
        bbox_img)
        print(f"{img_name} is done")
    toc2 = time.time()
    print(f"Avg. time to process {len(imgs)} imgs: {(toc2-toc1)/len(imgs)}")