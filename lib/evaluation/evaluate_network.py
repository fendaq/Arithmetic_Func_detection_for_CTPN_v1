import numpy as np
from lib.utils.bbox import bbox_overlaps, bbox_intersections

def evaluate_signal_proposal(p_bbox_list, g_bbox_list, thredshold):
    """
    计算单分类的网络性能
    :param g_bbox_list: groud truth shape: [n,4]
    :param p_bbox_list: predicted shape: [n,4] [left_top_x, left_top_y, right_bottom_x, right_bottom_y]
    :return
    """
    overlaps = bbox_overlaps(np.ascontiguousarray(p_bbox_list, dtype=np.float),
                             np.ascontiguousarray(g_bbox_list, dtype=np.float))
    # precision
    max_p_overlaps = np.max(overlaps, axis=1)
    # print(max_p_overlaps)
    filter = np.where(max_p_overlaps >= thredshold)
    # print(filter[0])

    precision_TP = len(filter[0])
    precision = float(precision_TP) / float(len(p_bbox_list))
    # print(precision, float(precision_TP), float(len(p_bbox_list)))

    # recall
    filted_overlaps = overlaps[filter[0]]
    # print(filted_overlaps)
    max_index = np.argmax(filted_overlaps, axis=1)
    # print(max_index.shape)

    recall = float(max_index.shape[0]) / float(len(g_bbox_list))

    return precision, recall


def evaluate_signal_bbox(p_bbox_list, g_bbox_list, thredshold):
    """
    计算单分类的网络性能
    :param p_bbox_list: predicted shape: [n,4] [left_top_x, left_top_y, right_bottom_x, right_bottom_y]
    :param g_bbox_list: groud truth shape: [n,4]
    :return
    """
    # print(g_bbox_list)
    overlaps = bbox_overlaps(np.ascontiguousarray(p_bbox_list, dtype=np.float),
                             np.ascontiguousarray(g_bbox_list, dtype=np.float),
                             1)
    max_overlaps = np.max(overlaps, axis=0)
    # print(max_overlaps)
    filter = np.where(max_overlaps>=thredshold)[0]

    recall = float(len(filter)) / float(len(g_bbox_list))

    return recall
#
# def _bbox_overlaps(p_bboxes, g_bboxes, is_signal_bbox=False):
#     p_bbox_num = np.shape(p_bboxes)[0]
#     g_bbox_num = np.shape(g_bboxes)[0]
#     overlaps = np.zeros([p_bbox_num, g_bbox_num])
#
#     for i in range(p_bbox_num):
#         for j in range(g_bbox_num):
#             overlaps[i][j] = iou(p_bboxes[i], g_bboxes[j], is_signal_bbox)
#     return overlaps
#
# def iou(p_bbox, g_bbox, is_signal_bbox=False):
#     x_gt = [g_bbox[0], g_bbox[2]]
#     y_gt = [g_bbox[1], g_bbox[3]]
#     x_test = [p_bbox[0], p_bbox[2]]
#     y_test = [p_bbox[1], p_bbox[3]]
#     x_gt.sort()
#     y_gt.sort()
#     x_test.sort()
#     y_test.sort()
#     if (x_gt[0] >= x_test[1] or y_gt[0] >= y_test[1] or x_gt[1] <= x_test[0] or y_gt[1] <= y_test[0]):
#         return 0
#     X = [max(x_gt[0], x_test[0]), min(x_gt[1], x_test[1])]
#     Y = [max(y_gt[0], y_test[0]), min(y_gt[1], y_test[1])]
#     cross_area = float(computeArea(X, Y))
#     gt_area = computeArea(x_gt, y_gt)
#     test_area = computeArea(x_test, y_test)
#     if is_signal_bbox:
#         return cross_area / gt_area
#     else:
#         return cross_area / (gt_area + test_area - cross_area)
#
#
# def computeArea(X, Y):
#     return abs(X[0] - X[1]) * abs(Y[0] - Y[1])


if __name__ == "__main__":
    p = [[0.880688, 0.44609185, 0.95696718, 0.6476958],
                     [0.84020283, 0.45787981, 0.99351478, 0.64294884],
                     [0.78723741, 0.61799151, 0.9083041, 0.75623035],
                     [0.22078986, 0.30151826, 0.36679274, 0.40551913],
                     [0.0041579, 0.48359361, 0.06867643, 0.60145104],
                     [0.4731401, 0.33888632, 0.75164948, 0.80546954],
                     [0.75489414, 0.75228018, 0.87922037, 0.88110524],
                     [0.21953127, 0.77934921, 0.34853417, 0.90626764],
                     [0.81, 0.11, 0.91, 0.21]]

    g = [[0.86132812, 0.48242188, 0.97460938, 0.6171875],
                   [0.18554688, 0.234375, 0.36132812, 0.41601562],
                   [0., 0.47265625, 0.0703125, 0.62109375],
                   [0.47070312, 0.3125, 0.77929688, 0.78125],
                   [0.8, 0.1, 0.9, 0.2]]

    precision, recall =evaluate(g,p)
    print(precision, recall)


