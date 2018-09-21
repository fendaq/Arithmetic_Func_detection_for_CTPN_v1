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
from lib.fast_rcnn.test import test_ctpn, run
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.text_connector.bbox_connector import BboxConnector


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



def ctpn(net, image_name):

    img = cv2.imread(image_name)
    #　将图像进行resize并返回其缩放大小
    img_resized, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)

    run_list, feed_dict = run_ctpn(net,img_resized )
    textdetector = TextDetector()
    # 得到是resize图像后的bbox
    text_proposals, scores, boxes = textdetector.detect(boxes, scores[:, np.newaxis], img_resized.shape[:2])

    # 原图像的绝对bbox位置
    original_bbox, scores = resize_bbox(boxes, scale)

    return

def run_ctpn(img):
    cfg_from_file('ctpn/text.yml')
    config = tf.ConfigProto(allow_soft_placement=True)
    # load network 构建网络模型
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    # 　将图像进行resize并返回其缩放大小
    img_resized, bbox_scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    # print(scale)
    run_list, feed_dict, im_scales = run(net, img_resized)

    return config, run_list, feed_dict, img_resized.shape, im_scales, bbox_scale

def decode_ctpn_output(ctpn_output, im_scales, bbox_scale, img_resized_shape):
    rois = ctpn_output[0]

    scores = rois[:, 0]
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        # print(im_scales[0])
        boxes = rois[:, 1:5] / im_scales[0]

    textdetector = TextDetector()
    # 得到是resize图像后的bbox
    text_proposals, scores, resized_boxes = textdetector.detect(boxes, scores[:, np.newaxis], img_resized_shape[:2])
    # 原图像的绝对bbox位置
    original_bbox, scores = resize_bbox(resized_boxes, bbox_scale)
    bbox_connector = BboxConnector(original_bbox)
    res_bbox = bbox_connector.start()
    return res_bbox

if __name__ == "__main__":
    img = cv2.imread('/home/tony/ocr/Arithmetic_Func_detection_for_CTPN_v1/data/demo/1.JPG')

    config, run_list, feed_dict, img_resized_shape, im_scales, bbox_scale = run_ctpn(img)

    sess = tf.Session(config=config)
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    out_put = sess.run(run_list, feed_dict)

    res_output = decode_ctpn_output(out_put, im_scales, bbox_scale, img_resized_shape)
    # print(res_output)
    for bbox in res_output:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)

    cv2.imwrite('dwad.jpg',img)


