import numpy as np
import os, cv2, pickle, json
import matplotlib.pyplot as plt



output_fig_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/compare_features_052023/figures/fig3/chamfer_distance'
raw_frame_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/extract_video_features_052023/data/Tail/video_frames/human_generalization'
edges_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/extract_video_features_052023/output/Tail/vis_full_resized_refined_features_zero_incl/big_C_right_no/arm-hand/Canny-edges'
video_name_in_raw_frame = 'interlaced_EL_action_big_C_right_no_00163_528x320'
crop_bbox = [230, 60, 510, 320]  # x1, y1, x2, y2


# Plot 3 frames as example
frame_list = [30, 42, 65]
# for fi in frame_list:
#     grasp_raw_frame_file = os.path.join(raw_frame_folder, video_name_in_raw_frame.replace('action', 'G'), '{:06d}.png'.format(fi))
#     grasp_raw_frame = cv2.imread(grasp_raw_frame_file)
#     reach_raw_frame_file = os.path.join(raw_frame_folder, video_name_in_raw_frame.replace('action', 'R'), '{:06d}.png'.format(fi+11))
#     reach_raw_frame = cv2.imread(reach_raw_frame_file)
#     # Crop
#     grasp_raw_frame = grasp_raw_frame[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
#     reach_raw_frame = reach_raw_frame[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
#     # Save in fig3 folder
#     cv2.imwrite(os.path.join(output_fig_folder, 'grasp_frame_{:06d}.png'.format(fi)), grasp_raw_frame)
#     cv2.imwrite(os.path.join(output_fig_folder, 'reach_frame_{:06d}.png'.format(fi)), reach_raw_frame)
#     # Read edges frame
#     grasp_edge_frame_file = os.path.join(edges_folder, 'grasp_{:06d}.png'.format(fi))
#     grasp_edge_frame = cv2.imread(grasp_edge_frame_file)
#     print(grasp_edge_frame.shape)
#     reach_edge_frame_file = os.path.join(edges_folder, 'reach_{:06d}.png'.format(fi))
#     reach_edge_frame = cv2.imread(reach_edge_frame_file)
#     # Change color: black background to white, white lines to deepskyblue for reach, red for grasp
#     grasp_edge_image = np.ones_like(grasp_edge_frame) * 255
#     for i in range(grasp_edge_frame.shape[0]):
#         for j in range(grasp_edge_frame.shape[1]):
#             if np.all(grasp_edge_frame[i, j] == [255, 255, 255]):
#                 grasp_edge_image[i, j] = [0, 0, 255]
#                 # Make lines wider
#                 if i > 0 and np.all(grasp_edge_frame[i-1, j] == [0, 0, 0]):
#                     grasp_edge_image[i-1, j] = [0, 0, 255]
#                 if i < grasp_edge_frame.shape[0]-1 and np.all(grasp_edge_frame[i+1, j] == [0, 0, 0]):
#                     grasp_edge_image[i+1, j] = [0, 0, 255]
#                 if j > 0 and np.all(grasp_edge_frame[i, j-1] == [0, 0, 0]):
#                     grasp_edge_image[i, j-1] = [0, 0, 255]
#                 if j < grasp_edge_frame.shape[1]-1 and np.all(grasp_edge_frame[i, j+1] == [0, 0, 0]):
#                     grasp_edge_image[i, j+1] = [0, 0, 255]
#     reach_edge_image = np.ones_like(reach_edge_frame) * 255
#     for i in range(reach_edge_frame.shape[0]):
#         for j in range(reach_edge_frame.shape[1]):
#             if np.all(reach_edge_frame[i, j] == [255, 255, 255]):
#                 reach_edge_image[i, j] = [255, 69, 0]
#                 # Make lines wider
#                 if i > 0 and np.all(reach_edge_frame[i-1, j] == [0, 0, 0]):
#                     reach_edge_image[i-1, j] = [255, 69, 0]
#                 if i < reach_edge_frame.shape[0]-1 and np.all(reach_edge_frame[i+1, j] == [0, 0, 0]):
#                     reach_edge_image[i+1, j] = [255, 69, 0]
#                 if j > 0 and np.all(reach_edge_frame[i, j-1] == [0, 0, 0]):
#                     reach_edge_image[i, j-1] = [255, 69, 0]
#     # Crop
#     grasp_edge_image = grasp_edge_image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
#     reach_edge_image = reach_edge_image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
#     # Save in fig3 folder
#     cv2.imwrite(os.path.join(output_fig_folder, 'grasp_edges_frame_{:06d}.png'.format(fi)), grasp_edge_image)
#     cv2.imwrite(os.path.join(output_fig_folder, 'reach_edges_frame_{:06d}.png'.format(fi)), reach_edge_image)
#
# # Plot 3D grasp points in orangered, reach points in royalblue
# fig = plt.figure(figsize=(5, 5), dpi=100)
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(azim=338, elev=22, roll=-100)
# ax.scatter(chd_points['grasp']['x'], chd_points['grasp']['y'], chd_points['grasp']['t'], c='orangered', marker='.', s=20, alpha=1)
# ax.scatter(chd_points['reach']['x'], chd_points['reach']['y'], chd_points['reach']['t'], c='royalblue', marker='.', s=20, alpha=1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('t')
# ax.set_zlim(t_range[0]-5, t_range[1]+5)
# ax.grid(True)
# plt.show()


# Plot entire video point set (downsampled)
feature_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/extract_video_features_052023/output/Tail/vis_full_resized_refined_features_zero_incl/big_C_right_no/arm-hand/Canny-edges'
grasp_frames = {}
reach_frames = {}
# grasp_raw_frames = {}
# reach_raw_frames = {}
for file in os.listdir(feature_folder):
    if 'grasp' in file:
        grasp_frame = cv2.imread(os.path.join(feature_folder, file))
        # Crop
        grasp_frame = grasp_frame[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
        # Downsample
        grasp_frame = cv2.resize(grasp_frame, (grasp_frame.shape[1]//2, grasp_frame.shape[0]//2))
        frame_index = int(file.split('_')[-1].split('.')[0])
        grasp_frames[frame_index] = grasp_frame
        # raw_frame = cv2.imread(os.path.join(raw_frame_folder, video_name_in_raw_frame.replace('action', 'G'), '{:06d}.png'.format(frame_index)))
        # # Crop
        # raw_frame = raw_frame[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
        # grasp_raw_frames[frame_index] = raw_frame
    elif 'reach' in file:
        reach_frame = cv2.imread(os.path.join(feature_folder, file))
        # Crop
        reach_frame = reach_frame[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
        # Downsample
        reach_frame = cv2.resize(reach_frame, (reach_frame.shape[1]//2, reach_frame.shape[0]//2))
        frame_index = int(file.split('_')[-1].split('.')[0])
        reach_frames[frame_index] = reach_frame
        # raw_frame = cv2.imread(os.path.join(raw_frame_folder, video_name_in_raw_frame.replace('action', 'R'), '{:06d}.png'.format(frame_index+11)))
        # # Crop
        # raw_frame = raw_frame[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
        # reach_raw_frames[frame_index] = raw_frame
print('grasp video shape:', grasp_frames[30].shape)
print('reach video shape:', reach_frames[30].shape)
grasp_points = []
g_x_list = []
g_y_list = []
g_t_list = []
# for frame_index, frame in grasp_frames.items():
# for frame_index in range(28,61):
for frame_index in [x for x in range(26, 40)] + [x for x in range(40, 61, 1)]:
    frame = grasp_frames[frame_index]
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            if frame[y, x, 0] > 0:
                grasp_points.append((x, y, frame_index))
                g_x_list.append(x)
                g_y_list.append(y)
                g_t_list.append(frame_index)
reach_points = []
r_x_list = []
r_y_list = []
r_t_list = []
# for frame_index, frame in reach_frames.items():
# for frame_index in range(28,61):
for frame_index in [x for x in range(26, 40)] + [x for x in range(40, 61, 1)]:
    frame = reach_frames[frame_index]
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            if frame[y, x, 0] > 0:
                reach_points.append((x, y, frame_index))
                r_x_list.append(x)
                r_y_list.append(y)
                r_t_list.append(frame_index)
print('grasp video num points:', len(g_x_list), len(g_y_list), len(g_t_list))
print('reach video num points:', len(r_x_list), len(r_y_list), len(r_t_list))
# Grasp
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(g_x_list, g_y_list, g_t_list, c='orangered', marker='o', s=1, edgecolors='none', alpha=0.5)
ax.view_init(azim=-18, elev=328, roll=-80)
# The other way: without reverse z, azim=338, elev=22, roll=-100
# Reverse z-axis
ax.invert_zaxis()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('t')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(output_fig_folder, 'grasp_3D_scatter.eps'), bbox_inches='tight', format='eps', transparent=True)
# Reach
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(r_x_list, r_y_list, r_t_list, c='royalblue', marker='o', s=1, edgecolors='none', alpha=0.5)
ax.view_init(azim=-18, elev=328, roll=-80)
# The other way: without reverse z, azim=338, elev=22, roll=-100
# Reverse z-axis
ax.invert_zaxis()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('t')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig(os.path.join(output_fig_folder, 'reach_3D_scatter.eps'), bbox_inches='tight', format='eps', transparent=True)

