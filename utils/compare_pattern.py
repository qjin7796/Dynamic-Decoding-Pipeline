# Functions to compute distance among video/image features.
# @Qiuhan Jin, 15/08/2023


from utils_feature_extraction import *
from utils_feature_visualization import *
import torch
from pykeops.torch import LazyTensor
from scipy.spatial.distance import squareform, correlation
from itertools import combinations



def construct_feature_point_set(list_of_video_feature_dict, feature_list, 
                                zero_incl=True, reduce_t=False, reduce_xy=False, 
                                output_folder=None, output_prefix=None, output_suffix=None, mask_label=None):
    """ Construct a point set of shape (num_points, num_dimensions) for each feature in each video,
        num_points = total number of points in the point set, 
        num_dimensions = total number of coordinates to describe a point,
            e.g. Global Sobel gradients: num_dimensions = 5 (x, y, frame, gradient_x, gradient_y),
                                         num_points = num_frames * num_pixels_in_frame.
        point coordinates are normalized across videos using min-max normalization.

        reduce_t: if True, reduce the temporal dimension by averaging over all frames.
        reduce_xy: if True, reduce the spatial dimension by averaging over all pixels.
    """
    list_of_video_feature_point_set = {}
    point_set_info = {}
    for feature_label in feature_list:
        feature_name = f'feature-{mask_label}-{feature_label}'
        print(datetime.now(), '---- Constructing point set of', feature_label, 'from input dict', feature_name)
        print(datetime.now(), '------ zero_incl', zero_incl, 'reduce_t', reduce_t, 'reduce_xy', reduce_xy)
        if feature_label in ['luminance', 'contrast', 'size', 'motion-degree', 'velocity', 'center', 'shape-moments']:
            # t, feature_res
            tmp_point_set = {}
            for video_name, video_feature_dict in list_of_video_feature_dict.items():
                tmp_point_set[video_name] = []
                for frame_index, feature_result in video_feature_dict[feature_name].items():
                    if feature_label in ['luminance', 'contrast', 'size', 'motion-degree', 'velocity']:
                        # feature_result is a scalar
                        tmp_coord = [frame_index, feature_result]
                    elif feature_label in ['center']:
                        # feature_result is a 2D np.array (1,2)
                        tmp_coord = [frame_index, feature_result[0], feature_result[1]]
                    elif feature_label in ['shape-moments']:
                        # feature_result is a 2D np.array (7,1)
                        tmp_coord = [frame_index] + feature_result.ravel().tolist()
                    tmp_point_set[video_name].append(tuple(tmp_coord))  # list of tuples
        elif feature_label in ['FAST-corners', 'shape']:
            # x, y, t if feature_res != 0, dimensions cannot be reduced
            tmp_point_set = {}
            for video_name, video_feature_dict in list_of_video_feature_dict.items():
                # num_points = len(video_feature_dict[feature_name] > 0)
                # Vectorized approach using np.where
                tmp_point_set[video_name] = []
                for frame_index, feature_result in video_feature_dict[feature_name].items():
                    if len(feature_result.shape) < 2:
                        continue
                    tmp_indices = np.where(feature_result != 0)
                    tmp_values = feature_result[tmp_indices]
                    for (frame_y, frame_x), value in zip(zip(*tmp_indices), tmp_values):
                        tmp_point_set[video_name].append((frame_x, frame_y, frame_index))  # list of tuples
        elif feature_label in ['Canny-edges']:  
            # x, y, t if feature_res != 0, dimensions can be reduced
            tmp_point_set = {}
            for video_name, video_feature_dict in list_of_video_feature_dict.items():
                # num_points = len(video_feature_dict[feature_name] > 0)
                # Vectorized approach using np.where
                if not reduce_t:
                    tmp_point_set[video_name] = []
                    for frame_index, feature_result in video_feature_dict[feature_name].items():
                        if len(feature_result.shape) < 2:
                            continue
                        tmp_indices = np.where(feature_result != 0)
                        tmp_values = feature_result[tmp_indices]
                        for (frame_y, frame_x), value in zip(zip(*tmp_indices), tmp_values):
                            tmp_point_set[video_name].append((frame_x, frame_y, frame_index))  # list of tuples
                else:
                    # Reduce temporal dimension
                    mean_feature_result = np.zeros_like(video_feature_dict[feature_name][list(video_feature_dict[feature_name].keys())[-1]], dtype=np.float32)
                    for frame_index, feature_result in video_feature_dict[feature_name].items():
                        try:
                            mean_feature_result += feature_result.astype(np.float32) / len(video_feature_dict[feature_name])
                        except ValueError:
                            if len(feature_result.shape) < 2:
                                continue
                            min_shape_0 = min(mean_feature_result.shape[0], feature_result.shape[0])
                            min_shape_1 = min(mean_feature_result.shape[1], feature_result.shape[1])
                            mean_feature_result[:min_shape_0, :min_shape_1] += feature_result.astype(np.float32)[:min_shape_0, :min_shape_1] / len(video_feature_dict[feature_name])
                        except:
                            print('!!! Error: Invalid feature shape -', feature_name)
                    tmp_point_set[video_name] = []
                    tmp_indices = np.where(mean_feature_result != 0)
                    tmp_values = mean_feature_result[tmp_indices]
                    for (frame_y, frame_x), value in zip(zip(*tmp_indices), tmp_values):
                        tmp_point_set[video_name].append((frame_x, frame_y))
        elif feature_label in ['Gabor-mean-response', 'HOG-image', 'flow', 'MHI', 'HOF-image', 'Sobel-gradients']:
            # x, y, t, feature_res if feature_res != 0, dimensions can be reduced
            tmp_point_set = {}
            for video_name, video_feature_dict in list_of_video_feature_dict.items():
                # num_points = len(video_feature_dict[feature_name]) * num_pixels > 0
                # Vectorized approach using np.where
                if not reduce_t and not reduce_xy:
                    tmp_point_set[video_name] = []
                    for frame_index, feature_result in video_feature_dict[feature_name].items():
                        if len(feature_result.shape) == 2:
                            # feature is 2D, i.e. only 1 value per pixel
                            if zero_incl:
                                # include zero features
                                tmp_indices = np.where(feature_result == 0)
                                tmp_values = feature_result[tmp_indices]
                                for (frame_y, frame_x), value in zip(zip(*tmp_indices), tmp_values):
                                    tmp_point_set[video_name].append((frame_x, frame_y, frame_index, value))  # list of tuples
                            tmp_indices = np.where(feature_result != 0)
                            tmp_values = feature_result[tmp_indices]
                            for (frame_y, frame_x), value in zip(zip(*tmp_indices), tmp_values):
                                tmp_point_set[video_name].append((frame_x, frame_y, frame_index, value))  # list of tuples
                        elif (len(feature_result.shape) == 3) and (feature_result.shape[-1] == 2):
                            # 3D, 3rd dim is 2 values
                            if zero_incl:
                                # include zero features
                                tmp_condition = np.logical_or(feature_result[..., 0] == 0, feature_result[..., 1] == 0)
                                tmp_indices = np.where(tmp_condition)
                                tmp_values = feature_result[tmp_indices]
                                for (frame_y, frame_x), [value_0, value_1] in zip(zip(*tmp_indices), tmp_values):
                                    tmp_point_set[video_name].append((frame_x, frame_y, frame_index, value_0, value_1))  # list of tuples
                            tmp_condition = np.logical_and(feature_result[..., 0] != 0, feature_result[..., 1] != 0)
                            tmp_indices = np.where(tmp_condition)
                            tmp_values = feature_result[tmp_indices]
                            for (frame_y, frame_x), [value_0, value_1] in zip(zip(*tmp_indices), tmp_values):
                                tmp_point_set[video_name].append((frame_x, frame_y, frame_index, value_0, value_1))  # list of tuples
                        else:
                            print('!!! Error: Invalid feature shape -', feature_name)
                elif reduce_t and not reduce_xy:
                    # Reduce temporal dimension
                    mean_feature_result = np.zeros_like(video_feature_dict[feature_name][list(video_feature_dict[feature_name].keys())[-1]], dtype=np.float32)
                    for frame_index, feature_result in video_feature_dict[feature_name].items():
                        try:
                            mean_feature_result += feature_result.astype(np.float32) / len(video_feature_dict[feature_name])
                        except ValueError:
                            if len(feature_result.shape) < 2:
                                continue
                            min_shape_0 = min(mean_feature_result.shape[0], feature_result.shape[0])
                            min_shape_1 = min(mean_feature_result.shape[1], feature_result.shape[1])
                            if len(mean_feature_result.shape) == 2:
                                mean_feature_result[:min_shape_0, :min_shape_1] += feature_result.astype(np.float32)[:min_shape_0, :min_shape_1] / len(video_feature_dict[feature_name])
                            else:
                                mean_feature_result[:min_shape_0, :min_shape_1, :] += feature_result.astype(np.float32)[:min_shape_0, :min_shape_1, :] / len(video_feature_dict[feature_name])
                        except:
                            print('!!! Error: Invalid feature shape -', feature_name)
                    tmp_point_set[video_name] = []
                    if len(mean_feature_result.shape) == 2:
                        if zero_incl:
                            # include zero features
                            tmp_indices = np.where(mean_feature_result == 0)
                            tmp_values = mean_feature_result[tmp_indices]
                            for (frame_y, frame_x), value in zip(zip(*tmp_indices), tmp_values):
                                tmp_point_set[video_name].append((frame_x, frame_y, value))
                        tmp_indices = np.where(mean_feature_result != 0)
                        tmp_values = mean_feature_result[tmp_indices]
                        for (frame_y, frame_x), value in zip(zip(*tmp_indices), tmp_values):
                            tmp_point_set[video_name].append((frame_x, frame_y, value))
                    elif (len(mean_feature_result.shape) == 3) and (mean_feature_result.shape[-1] == 2):
                        if zero_incl:
                            # include zero features
                            tmp_condition = np.logical_or(mean_feature_result[..., 0] == 0, mean_feature_result[..., 1] == 0)
                            tmp_indices = np.where(tmp_condition)
                            tmp_values = mean_feature_result[tmp_indices]
                            for (frame_y, frame_x), [value_0, value_1] in zip(zip(*tmp_indices), tmp_values):
                                tmp_point_set[video_name].append((frame_x, frame_y, value_0, value_1))
                        tmp_condition = np.logical_and(mean_feature_result[..., 0] != 0, mean_feature_result[..., 1] != 0)
                        tmp_indices = np.where(tmp_condition)
                        tmp_values = mean_feature_result[tmp_indices]
                        for (frame_y, frame_x), [value_0, value_1] in zip(zip(*tmp_indices), tmp_values):
                            tmp_point_set[video_name].append((frame_x, frame_y, value_0, value_1))
                    else:
                        print('!!! Error: Invalid feature shape -', feature_name)
                elif reduce_xy and not reduce_t:
                    # Reduce spatial dimension, i.e. compute mean over all pixels for each frame
                    tmp_point_set[video_name] = []
                    for frame_index, feature_result in video_feature_dict[feature_name].items():
                        if len(feature_result.shape) == 2:
                            tmp_point_set[video_name].append((frame_index, np.mean(feature_result)))
                        elif (len(feature_result.shape) == 3) and (feature_result.shape[-1] == 2):
                            tmp_point_set[video_name].append(
                                tuple([frame_index] + np.mean(feature_result, axis=(0,1)).tolist())
                            )
                        else:
                            print('!!! Error: Invalid feature shape -', feature_name)
        elif feature_label in ['2dGabor-energy']:
            # Gabor energy feature for each pixel in each frame is a 2D vector
            # The 2 dimensions are the parameters of the Gabor filters that contribute to the energy feature:
            #     'wavelength': [2,4,5,6,7,8,10],  # 7 values
            #     'orientation': np.arange(0, 195, 15),  # 0-180, 13 values
            # So altogether, the point set is 6D: x, y, t, w, o, energy, dimension can be reduced
            tmp_point_set = {}
            for video_name, video_feature_dict in list_of_video_feature_dict.items():
                # num_points = len(video_feature_dict[feature_name]) * 96 * 96 (resized output image) * 7 * 13
                # Vectorized approach using np.where
                tmp_point_set[video_name] = []
                for frame_index, feature_result in video_feature_dict[feature_name].items():
                    for w_o_key, w_o_energy in feature_result.items():
                        w_o_key_split = w_o_key.split('_')
                        wavelength = int(w_o_key_split[1])
                        orientation = int(w_o_key_split[3])
                        if zero_incl:
                            # include zero features
                            tmp_indices = np.where(w_o_energy == 0)
                            tmp_values = w_o_energy[tmp_indices]
                            for (frame_y, frame_x), value in zip(zip(*tmp_indices), tmp_values):
                                tmp_point_set[video_name].append((frame_x, frame_y, frame_index, wavelength, orientation, value))  # list of tuples
                        tmp_indices = np.where(w_o_energy != 0)
                        tmp_values = w_o_energy[tmp_indices]
                        for (frame_y, frame_x), value in zip(zip(*tmp_indices), tmp_values):
                            tmp_point_set[video_name].append((frame_x, frame_y, frame_index, wavelength, orientation, value))  # list of tuples
        elif feature_label in ['HOG-array', 'HOF-array']:
            # cellx, celly, t, orientation vector, num_dimensions = 11, dimension can be reduced
            tmp_point_set = {}
            for video_name, video_feature_dict in list_of_video_feature_dict.items():
                if not reduce_t and not reduce_xy:
                    tmp_point_set[video_name] = []
                    for frame_index, feature_result in video_feature_dict[feature_name].items():
                        if len(feature_result.shape) < 2:
                            continue
                        for celly in range(feature_result.shape[0]):
                            for cellx in range(feature_result.shape[1]):
                                tmp_point_set[video_name].append(
                                    tuple([celly, cellx, frame_index] + feature_result[celly, cellx].tolist())
                                )  # list of tuples
                elif reduce_t and not reduce_xy:
                    mean_feature_result = np.zeros_like(video_feature_dict[feature_name][list(video_feature_dict[feature_name].keys())[-1]], dtype=np.float32)
                    for frame_index, feature_result in video_feature_dict[feature_name].items():
                        try:
                            mean_feature_result += feature_result.astype(np.float32) / len(video_feature_dict[feature_name])
                        except ValueError:
                            if len(feature_result.shape) < 2:
                                continue
                            min_shape_0 = min(mean_feature_result.shape[0], feature_result.shape[0])
                            min_shape_1 = min(mean_feature_result.shape[1], feature_result.shape[1])
                            mean_feature_result[:min_shape_0, :min_shape_1, :] += feature_result.astype(np.float32)[:min_shape_0, :min_shape_1, :] / len(video_feature_dict[feature_name])
                    tmp_point_set[video_name] = []
                    for celly in range(mean_feature_result.shape[0]):
                        for cellx in range(mean_feature_result.shape[1]):
                            tmp_point_set[video_name].append(
                                tuple([celly, cellx] + mean_feature_result[celly, cellx].tolist())
                            )
                elif reduce_xy and not reduce_t:
                    tmp_point_set[video_name] = []
                    for frame_index, feature_result in video_feature_dict[feature_name].items():
                        if len(feature_result.shape) < 2:
                            continue
                        mean_feature_result = np.mean(feature_result, axis=(0,1))
                        tmp_point_set[video_name].append(
                            tuple([frame_index] + mean_feature_result.tolist())
                        )
        # Transform feature point set by min-max normalization
        # full_point_set_array = np.array([item for sublist in tmp_point_set.values() for item in sublist])
        # min_values = np.min(full_point_set_array, axis=0)
        # max_values = np.max(full_point_set_array, axis=0)
        # normalized_point_set_array = (full_point_set_array - min_values) / (max_values - min_values)
        # Transform feature point set by standardization (mean-variance normalization)
        full_point_set_array = np.array([item for sublist in tmp_point_set.values() for item in sublist])
        mean_values = np.nanmean(full_point_set_array, axis=0)  # nan elements are ignored
        std_values = np.nanstd(full_point_set_array, axis=0) + 1e-6  # avoid division by zero
        normalized_point_set_array = (full_point_set_array - mean_values) / std_values  # nan elements are kept
        normalized_point_set_array = np.nan_to_num(normalized_point_set_array)  # nan elements replaced by 0
        # Reconstruct feature point set dict
        start_index = 0
        for video_name, list_of_tuples in tmp_point_set.items():
            num_tuples = len(list_of_tuples)
            end_index = start_index + num_tuples
            if video_name not in list_of_video_feature_point_set.keys():
                list_of_video_feature_point_set[video_name] = {}
            list_of_video_feature_point_set[video_name][feature_label] = normalized_point_set_array[start_index:end_index]
            start_index = end_index
        print(datetime.now(), '------ Normalized point set array shape', normalized_point_set_array.shape)
        point_set_info[feature_label] = {'shape': normalized_point_set_array.shape, 'mean': mean_values, 'std': std_values}
        # Print each coordinate's dtype, value range and if nan exists
        for i in range(normalized_point_set_array.shape[1]):
            print(datetime.now(), '------ Coordinate', i, normalized_point_set_array[:,i].dtype, 
                  np.min(normalized_point_set_array[:,i]), np.max(normalized_point_set_array[:,i]), 
                  np.sum(np.isnan(normalized_point_set_array[:,i])))
            point_set_info[feature_label][f'coord-{i}'] = {'dtype': str(normalized_point_set_array[:,i].dtype), 
                                                           'min': np.min(normalized_point_set_array[:,i]), 
                                                           'max': np.max(normalized_point_set_array[:,i]), 
                                                           'nan': int(np.sum(np.isnan(normalized_point_set_array[:,i])))}
    # Save to npz file
    if output_folder is not None:
        if output_prefix is not None:
            output_prefix = f'{output_prefix}_'
        if reduce_xy:
            if mask_label is None:
                mask_label = 'xy-reduced'
            else:
                mask_label = f'{mask_label}-xy-reduced'
        if reduce_t:
            if mask_label is None:
                mask_label = 't-reduced'
            else:
                mask_label = f'{mask_label}-t-reduced'
        if zero_incl:
            if mask_label is None:
                mask_label = 'zero-incl'
            else:
                mask_label = f'{mask_label}-zero-incl'
        else:
            if mask_label is None:
                mask_label = 'zero-excl'
            else:
                mask_label = f'{mask_label}-zero-excl'
        if output_suffix is not None:
            mask_label = f'{mask_label}_{output_suffix}'
        output_name = os.path.join(output_folder, f"{output_prefix}feature-{mask_label}_point-set.npz")
        np.savez(output_name, Results=list_of_video_feature_point_set)
        print(datetime.now(), 'Point set saved to', output_name)
        output_info_name = os.path.join(output_folder, f"{output_prefix}feature-{mask_label}_point-set-info.json")
        with open(output_info_name, 'w') as fp:
            json.dump(point_set_info, fp, cls=NumpyArrayEncoder, indent=4)
    return list_of_video_feature_point_set, point_set_info


# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Python compilation (slower)
# def batched_pairwise_dist(a, b):
#     x, y = a.double(), b.double()
#     bs, num_points_x, points_dim = x.size()
#     bs, num_points_y, points_dim = y.size()
#     xx = torch.pow(x, 2).sum(2)
#     yy = torch.pow(y, 2).sum(2)
#     zz = torch.bmm(x, y.transpose(2, 1))
#     rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
#     ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
#     P = rx.transpose(2, 1) + ry - 2 * zz
#     return P
# def distChamfer(a, b):
#     """
#     :param a: Pointclouds Batch x nul_points x dim
#     :param b:  Pointclouds Batch x nul_points x dim
#     :return:
#     -closest point on b of points from a
#     -closest point on a of points from b
#     -idx of closest point on b of points from a
#     -idx of closest point on a of points from b
#     Works for pointcloud of any dimension
#     """
#     ## KeOps implementation:
#     ## input a, b should have shape (batch_size=1, num_points, dim)
#     bs, num_points_x, points_dim = a.size()
#     bs, num_points_y, points_dim = b.size()
#     x_i = LazyTensor(a.view(num_points_x, bs, points_dim))
#     y_j = LazyTensor(b.view(bs, num_points_y, points_dim))
#     D_ij = ((x_i - y_j)**2).sum(dim=2)
#     s_ij = D_ij.min(dim=0).mean()
#     s_ji = D_ij.min(dim=1).mean()
#     # s_ij = np.min(D_ij, axis=0).mean()
#     # s_ji = np.min(D_ij, axis=1).mean()
#     loss = s_ij + s_ji
#     return loss.item()


# def compute_chamfer_distance(list_of_video_feature_point_set, video_order_list, feature_list, 
#                              zero_incl=True, reduce_t=False, reduce_xy=False, 
#                              output_folder=None, output_prefix=None, output_suffix=None, mask_label=None):
#     """ Compute chamfer distance between two 3D point sets.
#         (1) Add batch dimension to point sets, i.e. (num_points, num_dimensions) -> (1, num_points, num_dimensions)
#             point_set_1 = point_set_1.unsqueeze(0)
#         (2) Based on num_dimensions, choose the corresponding chamfer_distance function
#     """
#     video_chd_array_dict = {}
#     video_chd_list_dict = {}
#     num_videos = len(video_order_list)
#     for feature_name in feature_list:
#         print(datetime.now(), '-- Compute chamfer distance of', feature_name)
#         chd_array = {}
#         chd_list = []
#         # Prepare all video tensors on cuda device
#         video_tensor_set = torch.tensor(np.array([list_of_video_feature_point_set[video_name][feature_name] for video_name in video_order_list]), dtype=torch.float32).cuda()
#         # Loop from video 1 to 143
#         for video_i in range(num_videos-1):
#             video_name = video_order_list[video_i]
#             video_chd_list = []
#             tensor_i = video_tensor_set[video_i, :, :]  # 1, num_points, num_dimensions
#             x_num_points, x_num_dims = tensor_i.size()
#             x_i = LazyTensor(video_tensor.view(num_points, 1, num_dims))  # single tensor
#             for video_j in range(video_i+1, num_videos):
#                 tensor_j = video_tensor_set[tensor_j, :, :]  # num_videos-video_i-1, num_points, num_dimensions
#                 y_num_points, y_num_dims = tensor_j.size()
#                 y_j = LazyTensor(batch_tensors.view(1, num_points, num_dims))  # single tensor
#                 ## Compute chamfer distance
#                 D_ij = ((x_i - y_j)**2).sum(-1)
#                 loss_tensor = (D_ij.min(dim=1).mean(dim=1) + D_ij.min(dim=2).mean(dim=1))
#                 ## Send to cpu and append to list
#                 loss_value = loss_tensor.item()
#                 ## Write data
#                 video_chd_list.append(loss_value)
#             chd_array[video_name] = video_chd_list
#             chd_list.extend(video_chd_list)
#         video_chd_array_dict[feature_name] = chd_array
#         video_chd_list_dict[feature_name] = chd_list
#         print(datetime.now(), '---- Number of pairs', len(chd_list), 'chd min', min(chd_list), 'max', max(chd_list), 'mean', np.mean(chd_list))
#         # Save to npz file
#         if output_folder is not None:
#             if output_prefix is not None:
#                 output_prefix = f'{output_prefix}_'
#             if mask_label is None:
#                 output_middle_name = feature_name
#             else:
#                 output_middle_name = f'{mask_label}-{feature_name}'
#             if reduce_xy:
#                 output_middle_name = f'{output_middle_name}-xy-reduced'
#             if reduce_t:
#                 output_middle_name = f'{output_middle_name}-t-reduced'
#             if zero_incl:
#                 output_middle_name = f'{output_middle_name}-zero-incl'
#             else:
#                 output_middle_name = f'{output_middle_name}-zero-excl'
#             if output_suffix is not None:
#                 output_middle_name = f'{output_middle_name}_{output_suffix}'
#             output_array_name = os.path.join(output_folder, f"{output_prefix}feature-{output_middle_name}_chamfer-distance-pairs.npz")
#             output_list_name = os.path.join(output_folder, f"{output_prefix}feature-{output_middle_name}_chamfer-distance-list.json")
#             np.savez(output_array_name, Results=chd_array)
#             with open(output_list_name, 'w') as f:
#                 json.dump({ feature_name: chd_list }, f, cls=NumpyArrayEncoder, indent=4)
#             print(datetime.now(), 'Chamfer distance matrix saved to', output_list_name)
#     return video_chd_array_dict, video_chd_list_dict


def generate_mini_batch_indices(sample_size, sample_index, mini_batch_size):
    """ Given a list of objects of length sample_size and a sample of index sample_index,
        split the rest (index > sample_index) of objects into batches of size <= mini_batch_size.
        Return a list of tuples, each tuple is the start and end index (excl) of a mini batch.
    """
    num_objects_to_compare = sample_size - sample_index - 1
    mini_batch_indices = []
    # num_objects_to_compare % mini_batch_size >= 0 and < mini_batch_size
    for batch_i in range(num_objects_to_compare // mini_batch_size):
        mini_batch_indices.append((sample_index+1+batch_i*mini_batch_size, sample_index+1+(batch_i+1)*mini_batch_size))
    # if num_objects_to_compare % mini_batch_size > 0, put them in one last mini batch
    if num_objects_to_compare % mini_batch_size > 0:
        mini_batch_indices.append((sample_index+1+(num_objects_to_compare//mini_batch_size)*mini_batch_size, sample_size))
    return mini_batch_indices


def compute_batch_chamfer_distance(list_of_video_feature_point_set, video_order_list, feature_list, 
                                   zero_incl=True, reduce_t=False, reduce_xy=False, 
                                   output_folder=None, output_prefix=None, output_suffix=None, mask_label=None):
    """ Compute chamfer distance between two 3D point sets.
        (1) Add batch dimension to point sets, i.e. (num_points, num_dimensions) -> (1, num_points, num_dimensions)
            point_set_1 = point_set_1.unsqueeze(0)
        (2) Based on num_dimensions, choose the corresponding chamfer_distance function
    """
    video_chd_array_dict = {}
    video_chd_list_dict = {}
    num_videos = len(video_order_list)
    if output_folder is not None:
        if output_prefix is not None:
            output_video_name = f'{output_prefix}_'
        else:
            output_video_name = ''
        if mask_label is not None:
            output_class_name = f'-{mask_label}'
        else:
            output_class_name = ''
        # Open a log file
        output_log_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-chamfer-distance-batch-log.txt")
        print('Log file saved to', output_log_name)
        with open(output_log_name, "a") as log_file:
            log_file.write(f'{datetime.now()}, Start processing {len(video_order_list)} videos with {len(list_of_video_feature_point_set)} features.\n')
            log_file.write(f'{datetime.now()}, Output folder: {output_folder}.\n\n')
    else:
        print('No output folder specified, log file not saved.')
    for feature_name in feature_list:
        print(datetime.now(), '-- Compute chamfer distance of', feature_name)
        with open(output_log_name, "a") as log_file:
            log_file.write(f'{datetime.now()}, Feature: {feature_name}\n')
        chd_array = {}
        chd_list = []
        # Prepare all video tensors on cuda device
        video_tensor_set = [torch.tensor(list_of_video_feature_point_set[video_name][feature_name], dtype=torch.float32).cuda() for video_name in video_order_list]
        # Loop from video 1 to 143
        for video_i in range(num_videos-1):
            video_name = video_order_list[video_i]
            with open(output_log_name, "a") as log_file:
                log_file.write(f'{datetime.now()}, -- video {video_i}: {video_name}\n')
            video_tensor = video_tensor_set[video_i]  # (num_points, num_dimensions)
            x_num_points, x_num_dims = video_tensor.size()
            x_i = LazyTensor(video_tensor.view(x_num_points, 1, x_num_dims))  # single tensor
            # ## Batch processing: compare video_i with video_i+1 to n-1
            # mini_batch_indices = generate_mini_batch_indices(num_videos, video_i, mini_batch_size=8)
            mini_chd_list = []
            # for batch_i, (start_i, end_i) in enumerate(mini_batch_indices):
            #     mini_batch_tensors = video_tensor_set[start_i:end_i]  # (mini_batch_size or smaller, num_points, num_dimensions)
            #     ### Compute chamfer distance
            #     bs, num_points, num_dims = mini_batch_tensors.size()
            #     y_j = LazyTensor(mini_batch_tensors.view(bs, 1, num_points, num_dims))  # batch of tensors (batches, 1, num_points, num_dimensions)
            #     D_ij = ((x_i - y_j)**2).sum(-1)
            #     loss_tensor = (D_ij.min(dim=1).mean(dim=1) + D_ij.min(dim=2).mean(dim=1))
            #     ### Send to cpu and convert to numpy array
            #     loss_list = torch.flatten(loss_tensor).tolist()
            #     mini_chd_list.extend(loss_list)
            for video_j in range(video_i+1, num_videos):
                video_tensor_j = video_tensor_set[video_j]
                y_num_points, y_num_dims = video_tensor_j.size()
                y_j = LazyTensor(video_tensor_j.view(1, y_num_points, y_num_dims))  # single point set tensor (1, num_points, num_dimensions)
                D_ij = ((x_i - y_j)**2).sum(-1)
                loss_tensor = (D_ij.min(dim=0).mean(dim=0) + D_ij.min(dim=1).mean(dim=0))
                ### Send to cpu float
                loss_value = loss_tensor.item()
                mini_chd_list.append(loss_value)
            ## Write data
            chd_array[video_name] = mini_chd_list
            chd_list.extend(mini_chd_list)
            print(datetime.now(), '----', video_name, len(mini_chd_list), 'mini_chd_list min', min(mini_chd_list), 'max', max(mini_chd_list), 'mean', np.mean(mini_chd_list))
            with open(output_log_name, "a") as log_file:
                log_file.write(f'{datetime.now()}, -- compared with {len(mini_chd_list)} videos: min {min(mini_chd_list)}, max {max(mini_chd_list)}, mean {np.mean(mini_chd_list)}\n\n')
            ## Save in time
            if output_folder is not None:
                output_feature_chd_array_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-{feature_name}_chamfer-distance-batch-results.npz")
                output_feature_chd_list_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-{feature_name}_chamfer-distance-list.json")
                np.savez(output_feature_chd_array_name, Results=chd_array)
                with open(output_feature_chd_list_name, 'w') as f:
                    json.dump({ feature_name: chd_list }, f, cls=NumpyArrayEncoder, indent=4)
        video_chd_array_dict[feature_name] = chd_array
        video_chd_list_dict[feature_name] = chd_list
        # Save to npz file
        if output_folder is not None:
            output_feature_chd_array_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-{feature_name}_chamfer-distance-batch-results.npz")
            output_feature_chd_list_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-{feature_name}_chamfer-distance-list.json")
            np.savez(output_feature_chd_array_name, Results=chd_array)
            with open(output_feature_chd_list_name, 'w') as f:
                json.dump({ feature_name: chd_list }, f, cls=NumpyArrayEncoder, indent=4)
            print(datetime.now(), 'Chamfer distance matrix saved to', output_feature_chd_list_name)
    return video_chd_array_dict, video_chd_list_dict


# def compute_chamfer_distance(list_of_video_feature_point_set, video_order_list, feature_list, 
#                              zero_incl=True, reduce_t=False, reduce_xy=False, 
#                              output_folder=None, output_prefix=None, output_suffix=None, mask_label=None):
#     """ Compute chamfer distance between two 3D point sets.
#         (1) Add batch dimension to point sets, i.e. (num_points, num_dimensions) -> (1, num_points, num_dimensions)
#             point_set_1 = point_set_1.unsqueeze(0)
#         (2) Based on num_dimensions, choose the corresponding chamfer_distance function
#     """
#     video_feature_chd_dict = {}
#     video_name_pairs = list(combinations(video_order_list, 2))
#     for feature_name in feature_list:
#         print(datetime.now(), '---- Compute chamfer distance of', feature_name)
#         tmp_min_dist_vector = []
#         for video_name_1, video_name_2 in video_name_pairs:
#             point_set_1 = torch.tensor(list_of_video_feature_point_set[video_name_1][feature_name], dtype=torch.float32).unsqueeze(0).cuda()
#             point_set_2 = torch.tensor(list_of_video_feature_point_set[video_name_2][feature_name], dtype=torch.float32).unsqueeze(0).cuda()
#             dim_empty_set_1 = [int(x) == 0 for x in point_set_1.shape]
#             dim_empty_set_2 = [int(x) == 0 for x in point_set_2.shape]
#             if any(dim_empty_set_1) and any(dim_empty_set_2):
#                 # Both sets are empty
#                 pair_min_dist = 0
#             else:
#                 try:
#                     pair_min_dist = distChamfer(point_set_1, point_set_2)
#                 except:
#                     # One set is empty
#                     pair_min_dist = np.inf
#             tmp_min_dist_vector.append(pair_min_dist)
#             print(datetime.now(), '------', video_name_1, video_name_2, pair_min_dist)
#         video_feature_chd_dict[feature_name] = tmp_min_dist_vector
#     # Save to json file
#     if output_folder is not None:
#         if output_prefix is not None:
#             output_prefix = f'{output_prefix}_'
#         if reduce_xy:
#             if mask_label is None:
#                 mask_label = 'xy-reduced'
#             else:
#                 mask_label = f'{mask_label}-xy-reduced'
#         if reduce_t:
#             if mask_label is None:
#                 mask_label = 't-reduced'
#             else:
#                 mask_label = f'{mask_label}-t-reduced'
#         if zero_incl:
#             if mask_label is None:
#                 mask_label = 'zero-incl'
#             else:
#                 mask_label = f'{mask_label}-zero-incl'
#         else:
#             if mask_label is None:
#                 mask_label = 'zero-excl'
#             else:
#                 mask_label = f'{mask_label}-zero-excl'
#         if output_suffix is not None:
#             mask_label = f'{mask_label}_{output_suffix}'
#         output_name = os.path.join(output_folder, f"{output_prefix}feature-{mask_label}_chamfer-distance.json")
#         with open(output_name, 'w') as f:
#             json.dump(video_feature_chd_dict, f, cls=NumpyArrayEncoder, indent=4)
#         print(datetime.now(), 'Chamfer distance array saved to', output_name)
#     return video_feature_chd_dict


def update_chamfer_distance(list_of_video_feature_point_set, video_order_list, feature_list, 
                            reduce_t=False, reduce_xy=False, rescale_factor=None, zero_incl=True,
                            video_set_name=None, seg_name=None, output_path=None):
    """ Update specific feature_list chamfer distance between two 3D point sets.
        (1) Add batch dimension to point sets, i.e. (num_points, num_dimensions) -> (1, num_points, num_dimensions)
            point_set_1 = point_set_1.unsqueeze(0)
        (2) Based on num_dimensions, choose the corresponding chamfer_distance function
    """
    video_feature_chd_dict = compute_chamfer_distance(
        list_of_video_feature_point_set, video_order_list, feature_list, 
        reduce_t, reduce_xy, rescale_factor, zero_incl,
        None, None, None
    )
    # Save to npz file
    if output_path is not None:
        output_prefix = None
        if reduce_xy:
            if output_prefix is None:
                output_prefix = 'xy-reduced'
            else:
                output_prefix = f'{output_prefix}-xy-reduced'
        if rescale_factor is not None:
            if output_prefix is None:
                output_prefix = 'rescaled'
            else:
                output_prefix = f'{output_prefix}-rescaled'
        if reduce_t:
            if output_prefix is None:
                output_prefix = 't-reduced'
            else:
                output_prefix = f'{output_prefix}-t-reduced'
        if zero_incl:
            if output_prefix is None:
                output_prefix = 'zero-incl'
            else:
                output_prefix = f'{output_prefix}-zero-incl'
        if output_prefix is not None:
            output_prefix = f'{output_prefix}-'
        output_name = os.path.join(output_path, f"{video_set_name}_feature-{seg_name}_{output_prefix}chamfer-distance.json")
        # with open(output_name, 'w') as f:
        #     json.dump(video_feature_chd_dict, f, cls=NumpyArrayEncoder, indent=4)
        # print('Chamfer distance saved to', output_name, datetime.now())
        # Update chamfer distance json file
        with open(output_name, 'r') as f:
            video_feature_chd_dict_old = json.load(f)
        for feature_name in feature_list:
            video_feature_chd_dict_old[feature_name] = video_feature_chd_dict[feature_name]
        with open(output_name, 'w') as f:
            json.dump(video_feature_chd_dict_old, f, cls=NumpyArrayEncoder, indent=4)
    return video_feature_chd_dict


