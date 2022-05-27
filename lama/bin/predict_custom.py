#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import argparse
import logging
import os
import sys
import traceback
import time

dirname = os.path.dirname(os.path.abspath(__file__))
dirname = os.path.dirname(dirname)
sys.path.append(dirname)
# sys.path.append("/home/nash/Desktop/projects/fashion/lama")
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
from matplotlib import pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'
        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        with torch.no_grad():
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(
                    predict_config.outdir, 
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = move_to_device(default_collate([dataset[img_i]]), device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                res = cv2.imwrite('result23.png', cur_res)
                cv2.imwrite(cur_out_fname, cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(
            f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)

def infer(img, mask, device='cuda'):

    if isinstance(img, str):
        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        return infer(img,mask,device)
    
    elif isinstance(mask, str):
        mask = cv2.imread(mask, cv2.COLOR_BGR2GRAY)
        return infer(img,mask,device)
    
    elif not (isinstance(img, np.ndarray) and isinstance(mask, np.ndarray)):
        raise AssertionError(
            f"{type(img)} != np.float32 or {type(mask)} != np.float32"
            )

    model_dir = "/home/nash/Desktop/projects/fashion/lama/big-lama"
    train_config_path = os.path.join(model_dir,"config.yaml")
    train_config = OmegaConf.load(train_config_path)
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    
    checkpoint_path = "/home/nash/Desktop/projects/fashion/lama/big-lama/models/best.ckpt"
    # device = 'cuda'
    device = device
    
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    # img_path = "/home/nash/Desktop/projects/fashion/lama/single_test_img/zalwomtops-1570-abb7e109e188ceaadb3d709b3ce4bb36.png"
    # mask_path = "single_test_img/zalwomtops-1570-abb7e109e188ceaadb3d709b3ce4bb36_mask.png"
    
    img = img.astype(np.float32)
    img = img/255.0
    img = torch.from_numpy(img)
    img = img.permute(2,0,1).unsqueeze(0) #HWC to BCHW
    img = img.type(torch.FloatTensor).to(device)
    
    mask = mask.astype(np.float32)
    mask = mask/255.0
    mask = torch.from_numpy(mask).unsqueeze(-1)
    # print(mask.shape)
    mask = mask.permute(2,0,1).unsqueeze(0) #HWC to BCHW
    mask = mask.type(torch.FloatTensor).to(device)
    batch = {'image':img, 'mask':mask}
    batch['mask'] = (batch['mask'] > 0) * 1
    # print(predict_config)
    
    batch = model(batch)
    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    # cur_res = batch['predicted_image'][0].permute(1, 2, 0).detach().cpu().numpy()

    unpad_to_size = batch.get('unpad_to_size', None)
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    # cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('result2.png',cur_res)
    return cur_res
    # cv2.imshow('changed',cur_res)
    # cv2.waitKey()
    # plt.imshow(cur_res)
    # plt.show()
    # plt.savefig('result.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img")
    parser.add_argument("--mask")
    parser.add_argument("--device", default='cpu')

    # main()
    args = parser.parse_args()
    tic = time.time()
    infer(args.img, args.mask, args.device)
    toc = time.time()
    print(f"Time taken {(toc-tic)}")