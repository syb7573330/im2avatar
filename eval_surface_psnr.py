import glob
import h5py
import os
import numpy as np
import scipy as sp
from PIL import Image
from scipy import misc
from scipy.spatial.distance import cdist

def rgb2ycc(vol):
  """ vol: (n, 3)
  """
  r = vol[:, 0]
  g = vol[:, 1]
  b = vol[:, 2]
  y = 0.299*r + 0.587*g + 0.114*b
  cb = 128 -.168736*r -.331364*g + .5*b
  cr = 128 +.5*r - .418688*g - .081312*b
  return np.column_stack((y, cb, cr))

### Variable to adjust #####
threshold = 0.8 # for occupancy 
adj_alpha = 0.2 # for adjusting blending weights
cat_id = "02958343"
############################./data/

vol_dim = 64
im_dim = 128
gt_output_path = "./data/ShapeNetCore_im2avatar/{}".format(cat_id)
output_shape_path = "./output_shape/{}".format(cat_id)
output_color_path = "./output_color/{}".format(cat_id)

ids = glob.glob(os.path.join(output_shape_path, "*"))
ids = [os.path.basename(i) for i in ids]

# # #
avg_psnr_rgb = 0.0
avg_psnr_ycc = 0.0
count = 0

for id_ in ids:
  # Load ground truth volume
  gt_path = os.path.join(gt_output_path, id_, "models/model_normalized_{}.h5".format(vol_dim))
  f_gt = h5py.File(gt_path)
  data_gt = f_gt['data'][:]
  indices_gt = np.where((data_gt[:,:,:,0] > -0.5) == 1)

  # Load views
  views_path = os.path.join(gt_output_path, id_, "views/*.png")
  views_paths = glob.glob(views_path)
  views_paths.sort()

  # prediction
  pred_color_path_id = os.path.join(output_color_path, id_)
  pred_colors_paths = glob.glob(os.path.join(pred_color_path_id, "*.h5"))
  pred_colors_paths.sort()

  pred_shape_path_id = os.path.join(output_shape_path, id_)
  pred_shapes_paths = glob.glob(os.path.join(pred_shape_path_id, "*.h5"))
  pred_shapes_paths.sort()

  for idx in range(len(pred_shapes_paths)):
    tmp_color_path = pred_colors_paths[idx]
    tmp_shape_path = pred_shapes_paths[idx]
    tmp_view_path = views_paths[idx]

    f_shape = h5py.File(tmp_shape_path)
    shape_res = f_shape['data'][:] 
    indices_ = np.where((shape_res[:,:,:,0] >= threshold) == 1)

    f_color = h5py.File(tmp_color_path)
    color_res = f_color['data'][:] 
    im = sp.misc.imread(tmp_view_path) 
    im = im / 255.0 
    
    # Set the background color as the top left corner pixel
    background_clr = im[0,0,0]

    ######## Start calculation ###########
    tmp_data = color_res[indices_] 

    # # # regressed color
    regressed_clr = tmp_data[:, :3] 

    # # # confidence 
    conf = tmp_data[:, 3:4] # confidence for regressed color
    conf = 1 - conf # confidence for flow color
    conf[conf > adj_alpha] = 1.0
    conf[conf <= adj_alpha] /= adj_alpha 

    # # # flow
    flow_x = tmp_data[:, 4] * im_dim 
    flow_y = tmp_data[:, 5] * im_dim 
    x_ = flow_x.astype(np.int)
    y_ = flow_y.astype(np.int)
    x_[x_>im_dim-1] = im_dim-1
    y_[y_>im_dim-1] = im_dim-1
    flow_clr = im[y_, x_]

    # replace sampled background colors with foreground colors
    bg_indices = np.where(np.all(abs(flow_clr - background_clr)<1e-3, axis=-1)) 
    bg_pos = np.column_stack((y_[bg_indices], x_[bg_indices])) 

    im_fg_indices = np.where(np.any(abs(im - background_clr)>1e-3, axis=-1)) 
    im_fg_clr = im[im_fg_indices]
    im_fg_pos = np.array(im_fg_indices).T 

    dis_mat = cdist(bg_pos, im_fg_pos) 
    dis_mat_pos = np.argmin(dis_mat, axis=1)

    flow_clr[bg_indices] = im_fg_clr[dis_mat_pos]

    # # # blended color
    clr_pred = regressed_clr * (1 - conf) + flow_clr * conf


    # # # ground truth color for indices_
    indices_np = np.column_stack(indices_)
    indices_gt_np = np.column_stack(indices_gt)
    dis_mat = cdist(indices_np, indices_gt_np) 
    dis_mat_pos = np.argmin(dis_mat, axis=1)
    indices_gt_clr = tuple(indices_gt_np[dis_mat_pos].T)
    clr_gt = data_gt[indices_gt_clr] 

    # # # color normaizlizatrion
    clr_pred *= 255.0
    clr_gt *= 255.0

    # # # rgb
    mse_rgb_tmp = np.sum(np.square(clr_pred - clr_gt), axis=0)

    # # # ycc
    clr_pred_ycc = rgb2ycc(clr_pred)
    clr_gt_ycc = rgb2ycc(clr_gt)
    mse_ycc_tmp = np.sum(np.square(clr_pred_ycc - clr_gt_ycc), axis=0)

    # # # 
    avg_mse_rgb_tmp = np.sum(mse_rgb_tmp) / indices_[0].shape[0] / 3
    avg_psnr_rgb_tmp = 20*np.log10(255.0) - 10*np.log10(avg_mse_rgb_tmp)
    avg_psnr_rgb += avg_psnr_rgb_tmp

    avg_mse_ycc_tmp = np.sum(mse_ycc_tmp) / indices_[0].shape[0] / 3
    avg_psnr_ycc_tmp = 20*np.log10(255.0) - 10*np.log10(avg_mse_ycc_tmp)
    avg_psnr_ycc += avg_psnr_ycc_tmp

    print "<Instance {}> psnr_rgb: {} psnr_ycc: {}".format(count / len(pred_shapes_paths), avg_psnr_rgb_tmp, avg_psnr_ycc_tmp)

    count += 1

print "Avg psnr rgb: {}".format(avg_psnr_rgb / float(count) )
print "Avg psnr ycc: {}".format(avg_psnr_ycc / float(count) )












