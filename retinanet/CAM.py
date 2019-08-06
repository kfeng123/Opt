"""
draw class activate mapping on retinanet
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.test_retinanet import _create_cell_anchors
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import cPickle as pickle
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

from caffe2.python import core, workspace


import pdb


c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default="./resource/configs/retinanet_R-101-FPN_1x_pose_bg.yaml",
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default="./resource/weights/new.pkl",
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='./test/out',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--im_or_folder', help='image or folder of images', default='./test/in'
    )
    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    return parser.parse_args()

"""
Find the box with the largest score
inputs:
    cls_boxes: a dictionary of length NUM_CLASSES;
                cls_boxes[i][:, :4] -- boxes (xmin, ymin, xmax, ymax)
                cls_boxes[i][:, 4] -- scores
outputs:
    max_loc: is the class of the target box
    cls_boxes[max_loc][j]: box and the score of the target box

"""
def location(cls_boxes):
    max_loc=0
    max_loc_score=0.0
    for i in range(len(cls_boxes)):
        if len(cls_boxes[i])==0:
            continue
        if max_loc_score < max(cls_boxes[i][:,4]):
            max_loc_score=max(cls_boxes[i][:,4])
            max_loc=i
    j=np.where(cls_boxes[max_loc]==max_loc_score)[0]
    return max_loc,cls_boxes[max_loc][j]


"""
If cls_box is a row of box_scores
inputs:
    box_scores: an array of shape (1000, 5). box_scores[:, :4]
                box_scores[:, :4] -- boxes (xmin, ymin, xmax, ymax)
                box_scores[:, 4] -- scores
    cls_box: one box and its score, shape (5, ).
outputs:
    boxes_all_loc: if cls_box is not a row of box_scores, return 6000.
                   Else return the row number of box_scores which equals cls_box.

"""
def judge_exist(box_scores,cls_box):
    boxes_all_loc=cfg.MODEL.NUM_CLASSES*1000
    for i in range(len(box_scores)):
        if (cls_box==box_scores[i]).all():
            boxes_all_loc=i
    return boxes_all_loc


"""
This function is adapted from detectron/core/test_retinanet.py:im_detect_bbox()
"""
def get_parameters(im,model,cls_boxes):
    anchors = _create_cell_anchors()
    # k_max = 7, k_min = 3
    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL
    # A = 9
    A = cfg.RETINANET.SCALES_PER_OCTAVE * len(cfg.RETINANET.ASPECT_RATIOS)
    inputs = {}
    # shape(inputs['data']): (1, 3, 768, 1408)  # Caffe2's convention: (N, C, H, W)
    inputs['data'], im_scale, inputs['im_info'] = \
        blob_utils.get_image_blob(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    cls_probs, box_preds = [], []
    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        cls_probs.append(core.ScopedName('retnet_cls_prob_{}'.format(suffix)))
        box_preds.append(core.ScopedName('retnet_bbox_pred_{}'.format(suffix)))
    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v.astype(np.float32, copy=False))

    workspace.RunNet(model.net.Proto().name)
    cls_probs = workspace.FetchBlobs(cls_probs)
    box_preds = workspace.FetchBlobs(box_preds)
    """
    cls_probs is a list of length:
    len(cls_probs) == cfg.MODEL.NUM_CLASSES - 1 == 5

    cls_probs[0].shape = (1, 45 , 96, 176) # fpn3
    45 = 9 * 5, 5 == cfg.MODEL.NUM_CLASSES - 1, 9 is the anchor number
    cls_probs[1].shape = (1, 45 , 48, 88) # fpn4
    cls_probs[2].shape = (1, 45 , 24, 44) # fpn5
    cls_probs[3].shape = (1, 45 , 12, 22) # fpn6
    cls_probs[4].shape = (1, 45 , 6, 11) # fpn7
    """
    """
    box_preds is a list of length:
    len(box_preds) == cfg.MODEL.NUM_CLASSES - 1 == 5

    box_preds[0].shape = (1, 36 , 96, 176) # fpn3
    45 = 9 * 4, 4 is the length of (xmin, ymin, xmax, ymax), 9 is the anchor number
    box_preds[1].shape = (1, 36 , 48, 88) # fpn4
    box_preds[2].shape = (1, 36 , 24, 44) # fpn5
    box_preds[3].shape = (1, 36 , 12, 22) # fpn6
    box_preds[4].shape = (1, 36 , 6, 11) # fpn7
    """


##################################
    # target class and its box
    cls_to_find,box_to_find = location(cls_boxes)
    box_to_find=box_to_find.ravel()

    # cls_to_find is a class number, e.g., 2.
    # box_to_find.shape ==5   (xmin, ymin, xmax, ymax)
##################################

    cnt = 0
    fpn_x = 0
    target_classes=0
    target_anchor_ids=0
    target_y=0
    target_x=0
    for lvl in range(k_min, k_max + 1):
        stride = 2. ** lvl
        cell_anchors = anchors[lvl]

        cls_prob = cls_probs[cnt]
        box_pred = box_preds[cnt]
        # if lvl == 7, cls_prob.shape == (1,45,6,11)
        cls_prob = cls_prob.reshape((
            cls_prob.shape[0], A, int(cls_prob.shape[1] / A),
            cls_prob.shape[2], cls_prob.shape[3]))
        # if lvl == 7, cls_prob.shape == (1,9,5,6,11)
        box_pred = box_pred.reshape((
            box_pred.shape[0], A, 4, box_pred.shape[2], box_pred.shape[3]))
        # if lvl == 7, box_pred.shape == (1,9,4,6,11)
        cnt += 1

        if cfg.RETINANET.SOFTMAX:
            cls_prob = cls_prob[:, :, 1::, :, :]

        # cls_prob_ravel.shape == (2970, )
        cls_prob_ravel = cls_prob.ravel()
        # th = 0.05
        th = cfg.RETINANET.INFERENCE_TH if lvl < k_max else 0.0
        # candidate_inds is the location of candidate_inds in cls_prob_ravel
        candidate_inds = np.where(cls_prob_ravel > th)[0]
        # generally, len(candidate_inds) > 0 only if lvl == 7
        if (len(candidate_inds) == 0):
            continue

        # typically, pre_nms_topn = 1000
        pre_nms_topn = min(cfg.RETINANET.PRE_NMS_TOP_N, len(candidate_inds))

        # inds is the locations of the largest 1000 candidate in candidate_inds
        inds = np.argpartition(
            cls_prob_ravel[candidate_inds], -pre_nms_topn)[-pre_nms_topn:]
        # now inds is the locations of the largest 1000 candidates in cls_prob_ravel
        # that is, cls_prob_ravel[inds[i]]  is a target probability
        inds = candidate_inds[inds]

        # inds_5d is the locations of candidates in cls_prob
        # that is, cls_prob[inds_5d[0], inds_5d[1], inds_5d[2], inds_5d[3], inds_5d[4]] is a target probability
        # inds_5d[0] is the N of the target
        # inds_5d[1] is the anchor id of the target
        # inds_5d[2] is the class number of the target
        # inds_5d[3] is the 'y' of the target
        # inds_5d[4] is the 'x' of the target
        inds_5d = np.array(np.unravel_index(inds, cls_prob.shape)).transpose()
        classes = inds_5d[:, 2]
        anchor_ids, y, x = inds_5d[:, 1], inds_5d[:, 3], inds_5d[:, 4]
        scores = cls_prob[:, anchor_ids, classes, y, x]


        boxes = np.column_stack((x, y, x, y)).astype(dtype=np.float32)
        boxes *= stride
        # the upper-left corner of each cell
        boxes += cell_anchors[anchor_ids, :]
        # anchors

        if not cfg.RETINANET.CLASS_SPECIFIC_BBOX:
            box_deltas = box_pred[0, anchor_ids, :, y, x]
        else:
            box_cls_inds = classes * 4
            box_deltas = np.vstack(
                [box_pred[0, ind:ind + 4, yi, xi]
                 for ind, yi, xi in zip(box_cls_inds, y, x)]
            )
        pred_boxes = (
            box_utils.bbox_transform(boxes, box_deltas)
            if cfg.TEST.BBOX_REG else boxes)
        pred_boxes /= im_scale
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        box_scores = np.zeros((pred_boxes.shape[0], 5))
        box_scores[:, 0:4] = pred_boxes
        box_scores[:, 4] = scores

############################################
        index = judge_exist(box_scores,box_to_find)
############################################
        if index == cfg.MODEL.NUM_CLASSES*1000:
            continue
        else:
            fpn_x=lvl
            target_classes=inds_5d[index,2]+1
            ######################################
            target_anchor_ids=inds_5d[index,1]
            # is it an error here?
            # it should be modified to
            # target_anchor_ids=inds_5d[index,1] + 1
            ######################################
            target_y=inds_5d[index,3]
            target_x=inds_5d[index,4]
    return fpn_x,target_classes,target_anchor_ids,target_y,target_x,pred_boxes[index]

def get_weight(model,classes,anchor_id):
    f=open(model)
    data=pickle.load(f)
    ind=(anchor_id-1)*(cfg.MODEL.NUM_CLASSES-1)+(classes-1)
    cls_weight=data['blobs']['retnet_cls_pred_fpn3_w']
    # cls_weight.shape:
    return cls_weight[ind]

def get_feature(fpn_x):
    suffix = 'fpn{}'.format(fpn_x)
    feature_map=workspace.FetchBlob(u'gpu_0/retnet_cls_conv_n3_{}'.format(suffix))
    return feature_map

def get_CAM(feature,weight,box):
    size_upsample=(int(box[3]-box[1]),int(box[2]-box[0]))
    output_cam=[]
    feature_tensor=torch.FloatTensor(feature)
    weight_tensor=torch.FloatTensor(weight).unsqueeze(0)
    result = F.conv2d(feature_tensor,weight_tensor,padding=1)
    # It seems that result should be added by 'gpu_0/retnet_cls_pred_fpn3_b' here?
    result = np.array(result.squeeze(1).squeeze(0))
    result=result-np.min(result)
    cam_img=result/np.max(result)
    cam_img=np.uint8(255*cam_img)
    output_cam.append(cv2.resize(cam_img,size_upsample))
    return output_cam


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_sitting_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        '''
        Get the location of the target (with largest cls_prob).
        Note that cls_boxes does NOT contain the location information.
        The largest probability lies in
        tmp = workspace.FetchBlob(u'gpu_0/retnet_cls_prob_fpn{}'.format(fpn_x))
        tmp.shape == (1,45,6,11)
        tmp[0,target_anchor_ids*5 + (target_classes-1), target_y, target_x]
        '''
        with c2_utils.NamedCudaScope(0):
            fpn_x,target_classes,target_anchor_ids,target_y,target_x,box=get_parameters(im,model,cls_boxes)
        target_weight = get_weight(args.weights,target_classes,target_anchor_ids)
        feature_map = get_feature(fpn_x)
        CAMs = get_CAM(feature_map,target_weight,box)
        height = int(box[3])-int(box[1])
        width = int(box[2])-int(box[0])
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        roi = im[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        dst = heatmap * 0.5 + roi * 0.3
        result = im.copy()
        result[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = dst
        str = dummy_coco_dataset['classes'][target_classes]
        cv2.imwrite(os.path.join(args.output_dir,"%s_%s_CAM.jpg"%(str,im_name.split('/')[-1])),result)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
