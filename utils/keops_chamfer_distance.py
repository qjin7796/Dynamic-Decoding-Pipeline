import numpy as np
import torch, os, json
from pykeops.torch import LazyTensor
from json import JSONEncoder
from datetime import datetime

class NumpyArrayEncoder(JSONEncoder):
    """ Special json encoder for numpy types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


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


def compute_batch_chamfer_distance(point_set_dict, point_set_name_list, point_set_feature_list, mini_batch_size=16,
                                   output_folder=None, output_prefix=None, output_label=None, save_log=False, test_mode=False):
    """ Compute chamfer distance between two multi-dimensional point sets.
        (1) Add batch dimension to point sets, i.e. (num_points, num_dimensions) -> (1, num_points, num_dimensions)
            point_set_1 = point_set_1.unsqueeze(0)
        (2) Based on num_dimensions, choose the corresponding chamfer_distance function
    """
    feature_batch_chd_array_dict = {}
    feature_chd_list_dict = {}
    num_videos = len(point_set_name_list)
    if output_folder is not None:
        if output_prefix is not None:
            output_video_name = f'{output_prefix}_'
        else:
            output_video_name = ''
        if output_label is not None:
            output_class_name = f'-{output_label}'
        else:
            output_class_name = ''
    if save_log:
        # Open a log file
        if output_folder is not None:
            output_log_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-chamfer-distance-batch-log.txt")
            print('Log file saved to', output_log_name)
            with open(output_log_name, "a") as log_file:
                log_file.write(f'{datetime.now()}, Start processing {len(point_set_name_list)} videos with {len(point_set_feature_list)} features.\n')
                log_file.write(f'{datetime.now()}, Batch size: {mini_batch_size}, Output folder: {output_folder}.\n\n')
        else:
            print('No output folder specified, log file not saved.')
    # Loop from feature 1 to n
    for feature_name in point_set_feature_list:
        print(datetime.now(), 'Processing feature', feature_name)
        if save_log:
            with open(output_log_name, "a") as log_file:
                log_file.write(f'{datetime.now()}, Feature: {feature_name}\n')
        batch_chd_array_dict = {}
        chd_list = []
        # Prepare all video tensors on cuda device
        video_tensor_set = [torch.tensor(point_set_dict[video_name][feature_name], dtype=torch.float32).cuda() for video_name in point_set_name_list]
        if test_mode:
            # Only compare the first video with the rest 143 videos
            num_videos_to_loop = 1
        else:
            num_videos_to_loop = num_videos - 1
        # Loop from video 1 to n-1
        for video_i in range(num_videos_to_loop):
            video_name = point_set_name_list[video_i]
            if save_log:
                with open(output_log_name, "a") as log_file:
                    log_file.write(f'{datetime.now()}, -- video {video_i}: {video_name}\n')
            video_tensor = video_tensor_set[video_i]  # (num_points, num_dimensions)
            x_num_points, x_num_dims = video_tensor.size()
            if mini_batch_size == 1:
                x_i = LazyTensor(video_tensor.view(x_num_points, 1, x_num_dims))  # single point set tensor (num_points, 1, num_dimensions)
            else:
                x_i = LazyTensor(video_tensor.view(1, x_num_points, 1, x_num_dims))  # single point set tensor in batch (1, num_points, 1, num_dimensions)
            # Compare video_i with video_i+1 to n-1
            if mini_batch_size == 1:
                ## Single processing: compare video_i with video_i+1 to n-1 one by one
                ## This applies to videos with different number of points
                mini_batch_chd_list = []
                # count = -1
                for video_j in range(video_i+1, num_videos):
                    # video_name_j = point_set_name_list[video_j]
                    video_tensor_j = video_tensor_set[video_j]
                    y_num_points, y_num_dims = video_tensor_j.size()
                    y_j = LazyTensor(video_tensor_j.view(1, y_num_points, y_num_dims))  # single point set tensor (1, num_points, num_dimensions)
                    D_ij = ((x_i - y_j)**2).sum(-1)
                    loss_tensor = (D_ij.min(dim=0).mean(dim=0) + D_ij.min(dim=1).mean(dim=0))
                    ### Send to cpu float
                    loss_value = loss_tensor.item()
                    mini_batch_chd_list.append(loss_value)
                    # count += 1
                    # if save_log and count == 0:
                    #     with open(output_log_name, "a") as log_file:
                    #         log_file.write(f'{datetime.now()}, ---- compared with video {video_j}: {video_name_j}, chamfer distance: {loss_value}\n')
            else:
                ## Batch processing: compare video_i with video_i+1 to n-1 in mini batches
                ## !!! Important: this only applies to videos with the same number of points !!!
                mini_batch_indices = generate_mini_batch_indices(num_videos, video_i, mini_batch_size)
                mini_batch_chd_list = []
                for batch_i, (start_i, end_i) in enumerate(mini_batch_indices):
                    mini_batch_tensors = torch.cat(video_tensor_set[start_i:end_i], 0)  # (mini_batch_size or smaller, num_points, num_dimensions)
                    ### Compute chamfer distance
                    bs, num_points, num_dims = mini_batch_tensors.size()
                    y_j = LazyTensor(mini_batch_tensors.view(bs, 1, num_points, num_dims))  # batch of tensors (batches, 1, num_points, num_dimensions)
                    D_ij = ((x_i - y_j)**2).sum(-1)
                    loss_tensor = (D_ij.min(dim=1).mean(dim=1) + D_ij.min(dim=2).mean(dim=1))
                    ### Send to cpu and convert to numpy array
                    loss_list = torch.flatten(loss_tensor).tolist()
                    mini_batch_chd_list.extend(loss_list)
                    if save_log:
                        with open(output_log_name, "a") as log_file:
                            log_file.write(f'{datetime.now()}, ---- batch {batch_i} video {start_i} to {end_i-1}: min {min(loss_list)}, max {max(loss_list)}, mean {np.mean(loss_list)}\n')
            ## Write data
            batch_chd_array_dict[video_name] = mini_batch_chd_list
            chd_list.extend(mini_batch_chd_list)
            # print(
            #     '--', video_name, 'compared with', len(mini_batch_chd_list), 'videos: min', 
            #     min(mini_batch_chd_list), 'max', max(mini_batch_chd_list), 'mean', np.mean(mini_batch_chd_list)
            # )
            ## Save and update batch results in time
            if output_folder is not None:
                output_feature_chd_array_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-{feature_name}_chamfer-distance-batch-results.npz")
                output_feature_chd_list_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-{feature_name}_chamfer-distance-list.json")
                np.savez(output_feature_chd_array_name, Results=batch_chd_array_dict)
                with open(output_feature_chd_list_name, 'w') as f:
                    json.dump({ feature_name: chd_list }, f, cls=NumpyArrayEncoder, indent=4)
            if save_log:
                with open(output_log_name, "a") as log_file:
                    log_file.write(f'{datetime.now()}, -- compared with {len(mini_batch_chd_list)} videos: min {min(mini_batch_chd_list)}, max {max(mini_batch_chd_list)}, mean {np.mean(mini_batch_chd_list)}\n\n')
        feature_batch_chd_array_dict[feature_name] = batch_chd_array_dict
        feature_chd_list_dict[feature_name] = chd_list
        if save_log:
            with open(output_log_name, "a") as log_file:
                log_file.write(f'{datetime.now()}, Feature {feature_name} done.\n Chamfer distance: min {min(chd_list)}, max {max(chd_list)}, mean {np.mean(chd_list)}\n\n\n')
        # Save to npz file
        if output_folder is not None:
            output_feature_chd_array_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-{feature_name}_chamfer-distance-batch-results.npz")
            output_feature_chd_list_name = os.path.join(output_folder, f"{output_video_name}feature{output_class_name}-{feature_name}_chamfer-distance-list.json")
            np.savez(output_feature_chd_array_name, Results=batch_chd_array_dict)
            with open(output_feature_chd_list_name, 'w') as f:
                json.dump({ feature_name: chd_list }, f, cls=NumpyArrayEncoder, indent=4)
            print('Chamfer distance list saved to', output_feature_chd_list_name)
    return feature_batch_chd_array_dict, feature_chd_list_dict

