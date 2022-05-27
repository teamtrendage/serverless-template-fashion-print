import argparse
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor
from tqdm import tqdm
from pytorch_lightning import Trainer

from datasets.coco import CocoDetection
from models.detr import Detr
from utils.visualization import drawbbox, visualize_predictions

from detr.datasets import get_coco_api_from_dataset
from detr.datasets.coco_eval import CocoEvaluator

def main(args):
    
  feature_extractor = DetrFeatureExtractor.from_pretrained(
    "facebook/detr-resnet-50")

  train_dataset = CocoDetection(
      img_folder='data/train/imgs',
      ann_file='data/train/train_coco.json',
      feature_extractor=feature_extractor)
  val_dataset = CocoDetection(
      img_folder='data/val/imgs',
      ann_file='data/val/val_coco.json',
      feature_extractor=feature_extractor)

  print("Number of training examples:", len(train_dataset))
  print("Number of validation examples:", len(val_dataset))


  # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
  image_ids = train_dataset.coco.getImgIds()
  # let's pick a random image
  image_id = image_ids[np.random.randint(0, len(image_ids))]
  # image_id = 23
  image = train_dataset.coco.loadImgs(image_id)[0]
  image = Image.open(image['file_name'])

  annotations = train_dataset.coco.imgToAnns[image_id]

  # drawbbox(image,annotations)

  def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values,
                                                          return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

  train_dataloader = DataLoader(
    train_dataset, collate_fn=collate_fn, batch_size=5, shuffle=True)
  val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
  batch = next(iter(train_dataloader))

  model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
              train_loader=train_dataloader,val_loader=val_dataloader)
  
  model = Detr.load_from_checkpoint(
      'checkpoints/lightning_logs/version_0/checkpoints/epoch=4-step=7755.ckpt',
      lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
      train_loader=train_dataloader,
      val_loader=val_dataloader
      )

  outputs = model(
    pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
  print(outputs.logits.shape)

  if args.train:
    trainer = Trainer(
      default_root_dir="./checkpoints", accelerator='gpu',
      max_epochs=10, gradient_clip_val=0.1
      )
    trainer.fit(model)

  if args.eval:
    base_ds = get_coco_api_from_dataset(val_dataset)

    iou_types = ['bbox']
    # initialize evaluator with ground truths
    coco_evaluator = CocoEvaluator(base_ds, iou_types) 

    model = Detr.load_from_checkpoint(
      'checkpoints/lightning_logs/version_1/checkpoints/epoch=9-step=15510.ckpt',
      lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
      train_loader=train_dataloader,
      val_loader=val_dataloader
      )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model.to(device)
    model.eval()

    print("Running evaluation...")

    for idx, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [
          {k: v.to(device) for k, v in t.items()} for t in batch["labels"]
          ] # these are in DETR format, resized + normalized

        # forward pass
        outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack(
          [target["orig_size"] for target in labels], dim=0
          )
        # convert outputs of model to COCO api
        results = feature_extractor.post_process(outputs, orig_target_sizes) 
        res = {
          target['image_id'].item(): output 
            for target, output in zip(labels, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

  # Inference
  if args.infer:
    model = Detr.load_from_checkpoint(
      'checkpoints/lightning_logs/version_0/checkpoints/epoch=4-step=7755.ckpt',
      lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
      train_loader=train_dataloader,
      val_loader=val_dataloader
      )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model.to(device)
    model.eval()

    pixel_values, target = val_dataset[1799]
    pixel_values = pixel_values.unsqueeze(0).to(device)
    print("IIIIIIIIIIIIIIII")
    print(pixel_values.shape)
    # forward pass to get class logits and bounding boxes
    outputs = model(pixel_values=pixel_values, pixel_mask=None)
    print("OOOOOOOOOOOOOOOOOO")
    # print(outputs)
    # print(outputs.shape)
    image_id = target['image_id'].item()
    image = val_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(image['file_name'])

    visualize_predictions(image, outputs, threshold=0.5)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", default=False, action='store_true')
  parser.add_argument("--eval", default=False, action='store_true')
  parser.add_argument("--infer", default=True, action='store_true')

  args = parser.parse_args()
  main(args)
