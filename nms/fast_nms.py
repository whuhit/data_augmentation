"""
https://github.com/dbolya/yolact/blob/57b8f2d95e62e2e649b382f516ab41f949b57239/layers/functions/detection.py
"""

def cc_fast_nms(self, boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200):
    # Collapse all the classes into 1
    scores, classes = scores.max(dim=0)

    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]

    boxes_idx = boxes[idx]

    # Compute the pairwise IoU between the boxes
    iou = jaccard(boxes_idx, boxes_idx)

    # Zero out the lower triangle of the cosine similarity matrix and diagonal
    iou.triu_(diagonal=1)

    # Now that everything in the diagonal and below is zeroed out, if we take the max
    # of the IoU matrix along the columns, each column will represent the maximum IoU
    # between this element and every element with a higher score than this element.
    iou_max, _ = torch.max(iou, dim=0)

    # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
    # don't have a higher scoring box that would supress it in normal NMS.
    idx_out = idx[iou_max <= iou_threshold]

    return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]


def fast_nms(self, boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200, second_threshold: bool = False):
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]

    num_classes, num_dets = idx.size()

    boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= iou_threshold)

    # We should also only keep detections over the confidence threshold, but at the cost of
    # maxing out your detection count for every image, you can just not do that. Because we
    # have such a minimal amount of computation per detection (matrix mulitplication only),
    # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
    # However, when you implement this in your method, you should do this second threshold.
    if second_threshold:
        keep *= (scores > self.conf_thresh)

    # Assign each kept detection to its corresponding class
    classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:cfg.max_num_detections]
    scores = scores[:cfg.max_num_detections]

    classes = classes[idx]
    boxes = boxes[idx]
    masks = masks[idx]

    return boxes, masks, classes, scores