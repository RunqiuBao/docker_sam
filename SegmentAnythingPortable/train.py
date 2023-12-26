from data import SAMDataset
from datasets import load_dataset
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import torch.nn.functional as F

from transformers import SamProcessor

import sys
sys.path.append('/root/data2/ftsam/')
from ftsam import FtSamDataset

if __name__ == "__main__":
#    dataset = load_dataset("nielsr/breast-cancer", split="train")
    dataset = FtSamDataset("train")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
      if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    num_epochs = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    model.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
          # forward pass
          outputs = model(pixel_values=batch["pixel_values"].to(device),
                          input_boxes=batch["input_boxes"].to(device),
                          multimask_output=False)
    
          # compute loss
          predicted_masks = outputs.pred_masks.squeeze(1)
          ground_truth_masks = batch["ground_truth_mask"].float().to(device)
          gtHeight, gtWidth = ground_truth_masks.shape[-2:]
          predicted_masks = F.interpolate(predicted_masks, size=(gtHeight, gtWidth))
#          from IPython import embed; embed()
          loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
    
          # backward pass (compute gradients of parameters w.r.t. loss)
          optimizer.zero_grad()
          loss.backward()
    
          # optimize
          optimizer.step()
          epoch_losses.append(loss.item())
    
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

