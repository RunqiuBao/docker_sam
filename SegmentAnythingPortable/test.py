from transformers import SamProcessor
from transformers import SamModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy
import argparse
from tqdm import tqdm
import monai
import os
import cv2
from statistics import mean

from data import SAMDataset

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset/'))
from ftsam import FtSamDataset


def SaveDebugImages(predictedMask, inputImage, savepath):
    inputImage = (inputImage - inputImage.min())
    inputImage = (inputImage * 255 / inputImage.max()).astype('uint8')
    bMask = cv2.threshold(predictedMask, 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8') * 255
    debugImage = cv2.resize(inputImage, (predictedMask.shape[1], predictedMask.shape[0]))
    debugImage[:, :, 0] = bMask
    cv2.imwrite(savepath, debugImage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('weightpath', help='path to the weight.')
    parser.add_argument('--debugpath', action='store', type=str, dest='debugpath', default='/root/logs/',
                        help='path to save the debug images. [default=%(default)s]')
    args = parser.parse_args()

    if not os.path.isdir(args.debugpath):
        os.makedirs(args.debugpath)

    promptType = 'points'

    # loading models, weights and datasets
    dataset = FtSamDataset("/root/data2/testsam/", "valid")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    test_dataset = SAMDataset(dataset=dataset, processor=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model = SamModel.from_pretrained("facebook/sam-vit-huge")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    checkpoint = torch.load(args.weightpath, map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage.cpu())
    model.load_state_dict(checkpoint['state_dict'])

    # prepare metrics
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    model.eval()
    with torch.no_grad():
        countBatch = 0
        epoch_losses = []
        for batch in tqdm(test_dataloader):
            # forward pass
            if promptType == "bbox":
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            elif promptType == "points":
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                          input_points=batch["input_points"].to(device),
                          input_labels=batch["input_labels"].to(device),
                          multimask_output=False)

            # compute loss
            SaveDebugImages(outputs.pred_masks.squeeze().cpu().numpy(), batch["pixel_values"].squeeze().cpu().permute(1, 2, 0).numpy(), os.path.join(args.debugpath, str(countBatch).zfill(6) + '.png'))
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            gtHeight, gtWidth = ground_truth_masks.shape[-2:]
            predicted_masks = F.interpolate(predicted_masks, size=(gtHeight, gtWidth))
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            epoch_losses.append(loss.item())
            countBatch += 1
        print(f"Mean loss test: {mean(epoch_losses)}")
