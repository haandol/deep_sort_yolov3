import logging
import mxnet as mx
from timeit import time
import warnings
import cv2
import numpy as np
import gluoncv as gcv
from gluoncv import model_zoo

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('deepsort')
warnings.filterwarnings('ignore')

times = {
    'det': 0.0,
    'enc': 0.0,
    'tot': 0.0
}

tooks = {
    'det': 0.0,
    'enc': 0.0,
    'tot': 0.0
}


def extend_bbox(bbox, ratio=.2):
    xmin, ymin, xmax, ymax = bbox
    return (
        max(0, xmin - xmin*ratio),
        max(0, ymin - ymin*ratio),
        xmax + xmax*ratio,
        ymax + ymax*ratio
    )


def main():
    logger.info('Start Tracking')
    ctx = mx.cpu()
    FPS = 13
    score_threshold = 0.5
    yolo = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
    yolo.reset_class(classes=['person'], reuse_weights=['person'])
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    capture = cv2.VideoCapture('video.mp4')
    frame_index = 0

    fps = 0.0

    t0 = time.time()
    while True:
        ret, frame = capture.read()
        if ret != True:
            break

        if 0 < FPS and frame_index % (30 // FPS) != 0:
            frame_index += 1
            continue

        boxs = []

        t2 = time.time()
        x, img = gcv.data.transforms.presets.yolo.transform_test(
            mx.nd.array(frame).astype('uint8'),
            short=416
        )
        x = x.as_in_context(ctx)
        logger.info('Transform: {:2.8f}'.format(time.time() - t2))

        t1 = time.time()
        class_IDs, det_scores, det_boxes = yolo(x)
        logger.info('Yolo Duration: {:2.8f}'.format(time.time() - t1))
        tooks['det'] += time.time() - t1

        t1 = time.time()
        person = mx.nd.array([0])
        score_threshold = mx.nd.array([0.5])
        for i, class_ID in enumerate(class_IDs[0]):
            if class_ID == person and det_scores[0][i] >= score_threshold:
                boxs.append(det_boxes[0][i].asnumpy())
        logger.info('Filter #{} boxs: {:2.8f}'.format(len(boxs), time.time() - t1))
        tooks['det'] += time.time() - t1
        times['det'] += 1

        t1 = time.time()
        features = encoder(img, boxs)
        logger.info('Encoder: {:2.8f}'.format(time.time() - t1))
        tooks['enc'] += time.time() - t1
        times['enc'] += 1

        t1 = time.time()
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        logger.info('Generate Detection: {:2.8f}'.format(time.time() - t1))

        t1 = time.time()
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        logger.info('NMS: {:2.8f}'.format(time.time() - t1))

        # Call the tracker
        t1 = time.time()
        tracker.predict()
        logger.info('Tracker Predict: {:2.8f}'.format(time.time() - t1))

        t1 = time.time()
        tracker.update(detections)
        logger.info('Tracker Update: {:2.8f}'.format(time.time() - t1))

        logger.info('{} Frame: {:2.8f}\n'.format(frame_index, time.time() - t2))

        frame_index = frame_index + 1

        new_img = img.copy()
        for track in tracker.tracks:
            bbox = [max(0, int(x)) for x in track.to_tlbr()]
            if not track.is_confirmed() or track.time_since_update > 1:
                if track.time_since_update == 2:
                    e_bbox = extend_bbox(bbox, ratio=.2)
                    e_bbox = [int(x) for x in e_bbox]
                    cv2.imwrite(
                        'missed/{}-{}.jpg'.format(frame_index, track.track_id),
                        img[e_bbox[1]:e_bbox[3], e_bbox[0]:e_bbox[2]],
                    )
                    logger.info('Cropped missed: xmin-{}, ymin-{}, xmax-{}, ymax-{}, shape-{}'.format(
                        e_bbox[0], e_bbox[1], e_bbox[2], e_bbox[3], img.shape
                    ))
                logger.info('Skipped by time_since_update')
                continue

            logger.info('Frame #{} - Id: {}'.format(frame_index, track.track_id))
            cv2.rectangle(new_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(255,255,255), 2)
            cv2.putText(new_img, str(track.track_id),(bbox[0], bbox[1]+30),0, 5e-3 * 200, (0,255,0),2)

        cv2.imwrite('images/{}.jpg'.format(frame_index), new_img)
        cv2.imshow('', new_img)

        fps  = (fps + (1./(time.time()-t2))) / 2
        logger.info("fps= %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logger.info('Time Elapsed: {:4.8f}'.format(time.time() - t0))
    tooks['tot'] += time.time() - t0
    times['tot'] += 1

    logger.info('Missed obj: {}, Missed frame: {}'.format(tracker.missed_obj, tracker.missed_frame))

    logger.info(tooks, times)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
