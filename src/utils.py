import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def matching(pred_points, gt_points, match_threshold=26.24):
   # 使用 cdist 方法计算成对距离
    distances_cdist = cdist(pred_points, gt_points)

    # 使用最近邻匹配进行关键点匹配
    matched_indices_pred_cdist = []
    matched_indices_gt_cdist = []

    # 遍历每个预测点
    for i in range(len(pred_points)):
        # 找到当前预测点与所有真实点中最近的点
        nearest_gt_index_cdist = np.argmin(distances_cdist[i])
        nearest_distance_cdist = distances_cdist[i, nearest_gt_index_cdist]

        # 如果最近距离小于阈值，则认为是有效匹配
        if nearest_distance_cdist <= match_threshold:
            matched_indices_pred_cdist.append(i)
            matched_indices_gt_cdist.append(nearest_gt_index_cdist)

    # 从匹配中提取距离
    matched_distances_nn_cdist = distances_cdist[matched_indices_pred_cdist, matched_indices_gt_cdist]


    return matched_distances_nn_cdist, matched_indices_pred_cdist, matched_indices_gt_cdist

def get_points(json_file):
    points = []

    for item in json_file['annotations']:
        try:
            x = item['keypoint']['x']
            y = item['keypoint']['y']
            points.append([y, x])
        except:
            pass

    return points


def extract_centroids(heatmap, threshold=0.5):
    # 应用阈值
    thresholded_map = np.where(heatmap > threshold, 1, 0)

    # 标记连通区域
    labeled_map = label(thresholded_map)

    # 计算每个区域的重心
    centroids = [region.centroid for region in regionprops(labeled_map)]
    
    return centroids
