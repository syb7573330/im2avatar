import glob
import os
import numpy as np
import h5py

def evaluate_voxel_prediction(prediction, gt):
  """  The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""
  intersection = np.sum(np.logical_and(prediction,gt))
  union = np.sum(np.logical_or(prediction,gt))
  IoU = float(intersection) / float(union)
  return IoU


### Variable to adjust #####
threshold = 0.8 
cat_id = "02958343"
############################./data/

pred_output_path = "./output_shape/{}".format(cat_id)
gt_output_path = "./data/ShapeNetCore_im2avatar/{}".format(cat_id)
vol_dim = 64

# get id for test shapes
ids = glob.glob(os.path.join(pred_output_path, "*"))
ids = [os.path.basename(i) for i in ids]

total_iou = 0.0
view_ious = [0.0] * 12 # The IoU for each view
count = 0

for id_ in ids:
  # Load ground truth volume
  gt_path = os.path.join(gt_output_path, id_, "models/model_normalized_{}.h5".format(vol_dim))
  f_gt = h5py.File(gt_path)
  data_gt = f_gt['data'][:]
  shape_gt = data_gt[:,:,:,0] > -0.5
  shape_gt = shape_gt.astype(np.int)

  # Load predicted shape
  pred_path = os.path.join(pred_output_path, id_)
  pred_shapes_paths = glob.glob(os.path.join(pred_path, "*.h5"))
  pred_shapes_paths.sort()

  for idx in range(len(pred_shapes_paths)):
    tmp_path = pred_shapes_paths[idx]

    f = h5py.File(tmp_path)
    data = f['data'][:]
    data = data[:,:,:,0]
    shape_pred = np.where(data > threshold, np.ones_like(data), np.zeros_like(data)) 
    shape_pred = shape_pred.astype(np.int)

    tmp_iou = evaluate_voxel_prediction(shape_pred, shape_gt)

    total_iou += tmp_iou
    view_ious[idx] += tmp_iou

    print "<Instance {}> iou: {}".format(count, tmp_iou)
  count += 1

 
print "avg iou: {}".format(total_iou / float(count) / 12.0)
for i in range(12):
  print "view iou {}: {}".format(i, view_ious[i] / float(count))