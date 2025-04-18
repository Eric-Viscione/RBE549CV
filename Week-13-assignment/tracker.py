##Code forked from https://github.com/computervisioneng/object-tracking-yolov8-deep-sort
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np
from random import randint
track_id_colors = {}
class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.6
        nn_budget = None

        encoder_model_filename = 'model_data/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric, n_init=6)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)
    def update(self, frame, detections):
        self.last_classes = {}
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return
        original_bboxes = [tuple(d[0:4]) for d in detections] 
        bboxes = np.asarray(original_bboxes)
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [float(d[4]) for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        self._last_detection_info = []

        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))
            original_bbox = np.array(original_bboxes[bbox_id])  # (x1, y1, x2, y2)
            class_id = int(detections[bbox_id][5])
            class_name = detections[bbox_id][6]
            self._last_detection_info.append((original_bbox, class_id, class_name, scores[bbox_id]))

            

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr().astype(int)  # (x1, y1, x2, y2)
            best_iou = 0
            best_class = (None, None)
            best_score = None
            for det_bbox, class_id, class_name, score in self._last_detection_info:
                iou = self._compute_iou(bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_class = (class_id, class_name)
                    best_score = score

            uuid = track.track_id
            class_id, class_name = best_class
            if uuid not in track_id_colors:
                track_id_colors[uuid] = (
                    randint(64, 255),
                    randint(64, 255),
                    randint(64, 255)
                )
            color = track_id_colors[uuid]
            tracks.append(Track(uuid, bbox, class_id, class_name, score, color))
            
        self.tracks = tracks

class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox, class_id=None, class_name=None, score=None, color=None):
        self.track_id = id
        self.bbox = bbox
        self.class_id = class_id 
        self.class_name = class_name
        self.has_crossed = False
        self.score = score
        self.bb_color = color