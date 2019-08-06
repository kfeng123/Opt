import numpy as np
from collections import defaultdict

from caffe2.python import core, workspace
from detectron.core.config import cfg

import pdb

# the following two lines solve the following problem
# AttributeError: Method UpsampleNearest is not a registered operator.
import detectron.utils.c2 as c2_utils
c2_utils.import_detectron_ops()

###################### testing new image ##############################

# import detectron.core.test_engine as infer_engine
import detectron.core.test_retinanet as test_retinanet
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

def simple_detect(model, im, myNMS = True):
    # adapted from detectron/core/test_retinanet.py:im_detect_bbox()

    anchors = test_retinanet._create_cell_anchors()
    # anchors are input independent and could be precomputed
    # anchors save the coordinates (relative to the upper-left of the cells) of the upper-left and the bottom-right of the bbox. (xmin, ymin, xmax, ymax)
    # anchors[3].shape = ... = anchors[7].shape = (9, 4)


    k_max, k_min = 7, 3
    A = 9
    inputs = {}
    inputs["data"], im_scale, inputs['im_info'] = \
            blob_utils.get_image_blob(im,
                    # cfg.TEST.SCALE
                    800
                    ,
                    # cfg.TEST.MAX_SIZE
                    1333
                    )
    # inputs['data'].shape == (1, 3, 768, 1408)   #BGR#


    cls_probs, box_preds = [], []
    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        cls_probs.append(core.ScopedName("retnet_cls_prob_{}".format(suffix)))
        box_preds.append(core.ScopedName("retnet_bbox_pred_{}".format(suffix)))
    # feed input to the workspace
    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v.astype(np.float32, copy=False))
    # run inference
    workspace.RunNet(model.net.Proto().name)
    # fetch inference results: cls_probs box_preds
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

    box_preds is a list of length:
    len(box_preds) == cfg.MODEL.NUM_CLASSES - 1 == 5

    box_preds[0].shape = (1, 36 , 96, 176) # fpn3
    45 = 9 * 4, 4 is the length of (xmin, ymin, xmax, ymax), 9 is the anchor number
    box_preds[1].shape = (1, 36 , 48, 88) # fpn4
    box_preds[2].shape = (1, 36 , 24, 44) # fpn5
    box_preds[3].shape = (1, 36 , 12, 22) # fpn6
    box_preds[4].shape = (1, 36 , 6, 11) # fpn7
    """


    # here the boxes_all are [x0, y0, x1, y1, score]
    boxes_all = defaultdict(list)

    cnt = 0
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = anchors[lvl]
        # cell_anchors.shape = (9,4)

        cls_prob = cls_probs[cnt]
        box_pred = box_preds[cnt]
        cls_prob = cls_prob.reshape((
                cls_prob.shape[0], A, int(cls_prob.shape[1] / A),
                cls_prob.shape[2], cls_prob.shape[3] ))
        box_pred = box_pred.reshape((
            box_pred.shape[0], A, 4, box_pred.shape[2], box_pred.shape[3]
            ))
        cnt += 1
        if cfg.RETINANET.SOFTMAX:
            cls_prob = cls_prob[:,:,1::,:,:]
        cls_prob_ravel = cls_prob.ravel()
        th = cfg.RETINANET.INFERENCE_TH if lvl < k_max else 0.0
        candidate_inds = np.where(cls_prob_ravel > th)[0]
        if (len(candidate_inds) == 0):
            continue

        pre_nms_topn = min(cfg.RETINANET.PRE_NMS_TOP_N, len(candidate_inds))
        inds = np.argpartition(
                cls_prob_ravel[candidate_inds], - pre_nms_topn
                )[-pre_nms_topn:]
        inds = candidate_inds[inds]

        inds_5d = np.array(np.unravel_index(inds, cls_prob.shape)).transpose()
        classes = inds_5d[:, 2]
        anchor_ids, y, x = inds_5d[:, 1], inds_5d[:, 3], inds_5d[:, 4]
        scores = cls_prob[:, anchor_ids, classes, y, x]


        boxes = np.column_stack((x, y, x, y)).astype(dtype=np.float32)
        boxes *= stride
        boxes += cell_anchors[anchor_ids, :]

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

        for cls in range(1, cfg.MODEL.NUM_CLASSES):
            inds = np.where(classes == cls - 1)[0]
            if len(inds) > 0:
                boxes_all[cls].extend(box_scores[inds, :])



    # Combine predictions across all levels and retain the top scoring by class
    detections = []
    for cls, boxes in boxes_all.items():
        cls_dets = np.vstack(boxes).astype(dtype=np.float32)
        # do class specific nms here

        # NMS
        #if cfg.TEST.SOFT_NMS.ENABLED:
        #    cls_dets, keep = box_utils.soft_nms(
        #        cls_dets,
        #        sigma=cfg.TEST.SOFT_NMS.SIGMA,
        #        overlap_thresh=cfg.TEST.NMS,
        #        score_thresh=0.0001,
        #        method=cfg.TEST.SOFT_NMS.METHOD
        #    )
        #else:
        #    keep = box_utils.nms(cls_dets, cfg.TEST.NMS)
        #    cls_dets = cls_dets[keep, :]
        if myNMS:
            keep = box_utils.nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]


        #out = np.zeros((len(keep), 6))
        out = np.zeros((cls_dets.shape[0], 6))

        out[:, 0:5] = cls_dets
        out[:, 5].fill(cls)
        detections.append(out)


    # detections (N, 6) format:
    #   detections[:, :4] - boxes
    #   detections[:, 4] - scores
    #   detections[:, 5] - classes
    detections = np.vstack(detections)

    # sort all again
    inds = np.argsort(-detections[:, 4])
    detections = detections[inds[0:cfg.TEST.DETECTIONS_PER_IM], :]

    # Convert the detections to image cls_ format (see core/test_engine.py)
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for c in range(1, num_classes):
        inds = np.where(detections[:, 5] == c)[0]
        cls_boxes[c] = detections[inds, :5]

    return cls_boxes



