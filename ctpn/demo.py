from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.evaluation.evaluate_network import evaluate_signal_proposal,evaluate_signal_bbox
from lib.prepare_training_data.parse_tal_xml import ParseXml




def resize_im(im, scale, max_scale=None):

    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])

    iiimg = cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    return iiimg, f

def resize_bbox(boxes, scale):
    resized_bbox = []
    scores = []
    for box in boxes:
        bbox = []
        bbox.append(min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)))
        bbox.append(min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)))
        bbox.append(max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)))
        bbox.append(max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)))
        scores.append(box[8])
        resized_bbox.append(bbox)

    return resized_bbox, scores


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    with open('data/results/' + '{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if box[4] >= 0.9:
                color = (0, 255, 0)
            elif box[4] >= 0.8:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[5])),color, 1)

            # min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            # min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            # max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            # max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(box[0]), str(box[1]), str(box[2]), str(box[5])]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)


def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    # print('111', img.shape)
    #　将图像进行resize并返回其缩放大小
    img_resized, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    # print('222', img.shape)
    # 送入网络得到1000个得分,1000个bbox
    scores, boxes = test_ctpn(sess, net, img_resized)


    textdetector = TextDetector()
    # 得到是resize图像后的bbox
    text_proposals, scores, boxes = textdetector.detect(boxes, scores[:, np.newaxis], img_resized.shape[:2])
    draw_boxes(img_resized, image_name, boxes, scale)

    # 原图像的绝对bbox位置
    original_bbox, scores = resize_bbox(boxes, scale)


    image_id = image_name.split('/')[-1]
    image_id = image_id.split('.')[0]

    is_evaluate_bbox = True
    if is_evaluate_bbox:
        test_label_dir = './data/test_xml'
        p = ParseXml(os.path.join(test_label_dir, image_id+'.xml'), rect=True)
        img_name, class_list, g_bbox_list = p.get_bbox_class()

        recall = evaluate_signal_bbox(original_bbox, g_bbox_list, 0.7)
        print('recall', recall)
        res = [recall]

        img_re = img
        for i in range(np.shape(original_bbox)[0]):
            cv2.rectangle(img_re, (original_bbox[i][0], original_bbox[i][1]),
                          (original_bbox[i][2], original_bbox[i][3]), (0, 255, 0), 1)

        for i in range(len(g_bbox_list)):
            cv2.rectangle(img_re, (g_bbox_list[i][0], g_bbox_list[i][1]),
                          (g_bbox_list[i][2], g_bbox_list[i][3]), (255, 0, 0), 1)
        # cv2.imshow('333', img_re)
        cv2.imwrite(os.path.join('./data/g_p_bbox', 'bb_'+image_id + '.jpg'), img_re)
        # cv2.waitKey()

    is_evaluate_proposal = False
    if is_evaluate_proposal:
        test_label_dir = './data/test_label'
        g_bbox = []
        with open(os.path.join(test_label_dir, image_id+'.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                _, x1, y1, x2, y2 = line.split()
                # print(_, x1, y1, x2, y2)
                g_bbox.append([int(x1), int(y1), int(x2), int(y2)])

        precision, recall = evaluate_signal_proposal(text_proposals, g_bbox, 0.5)
        print('precision', precision, 'recall', recall)
        img_re = img_resized
        for i in range(np.shape(np.array(g_bbox))[0]):
            cv2.rectangle(img_re, (g_bbox[i][0], g_bbox[i][1]),
                          (g_bbox[i][2], g_bbox[i][3]), (255, 0, 0), 1)
        cv2.imwrite(os.path.join('./data/g_p_proposal', image_id+'.jpg'), img_re)
        res = [recall, precision]

    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    return res

if __name__ == '__main__':
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network 构建网络模型
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
    #im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in range(2):
    #     _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.JPG'))
    total_recall = 0
    total_precision = 0
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        res = ctpn(sess, net, im_name)
        total_recall += res[0]
        if len(res)==2:
            total_precision += res[1]
    if total_precision > 0:
        print('average precision = ', float(total_precision) / len(im_names))
        print('average recall = ', float(total_recall) / len(im_names))
    else:
        print('average recall = ', float(total_recall)/len(im_names))
