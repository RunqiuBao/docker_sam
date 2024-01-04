import numpy
from utils import get_bounding_box, SampleRandomNPointsFromMask
from datasets import load_dataset
from torch.utils.data import Dataset


class SAMDataset(Dataset):
  _promptType = None

  dataset = None
  processor = None
  vRandomIndices = None
  numPoints = 7  # randomly sampled points prompt in each gt mask. hard coded param.

  def __init__(self, dataset, processor, promptType='points'):
    assert(promptType in ('points', 'bbox'))
    self.dataset = dataset
    # generate random indices for point prompts
    self.vRandomIndices = []
    for indexDatum in range(len(self.dataset)):
      ground_truth_mask = numpy.array(self.dataset[indexDatum]["label"])
      coordsMask = numpy.where(ground_truth_mask == 1)
      indicesRaveledMask = numpy.arange(coordsMask[0].shape[0])
      rng = numpy.random.default_rng()
      randomIndices = rng.choice(indicesRaveledMask, self.numPoints, replace=False)
      self.vRandomIndices.append(randomIndices)
    self.processor = processor
    self._promptType = promptType

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = numpy.array(item["label"])

    if self._promptType == 'bbox':
      # get bounding box prompt
      prompt = get_bounding_box(ground_truth_mask)
      # prepare image and prompt for the model
      inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
    elif self._promptType == 'points':
      input_points, input_labels = SampleRandomNPointsFromMask(ground_truth_mask, self.numPoints, self.vRandomIndices[idx])
      inputs = self.processor(image, input_points=[[input_points]], input_labels=[[input_labels]], return_tensors="pt")
#    from IPython import embed; embed()

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs
