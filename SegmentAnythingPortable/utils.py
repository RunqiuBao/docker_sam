import numpy


def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = numpy.where(ground_truth_map > 0)
  x_min, x_max = numpy.min(x_indices), numpy.max(x_indices)
  y_min, y_max = numpy.min(y_indices), numpy.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - numpy.random.randint(0, 20))
  x_max = min(W, x_max + numpy.random.randint(0, 20))
  y_min = max(0, y_min - numpy.random.randint(0, 20))
  y_max = min(H, y_max + numpy.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox
