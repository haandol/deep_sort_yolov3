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

def main():
    print('Start Tracking')
    ctx = mx.cpu()
    FPS = 15
    score_threshold = 0.5
    # yolo = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
    yolo = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    capture = cv2.VideoCapture('v1.mp4')
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
        print('Transform: {:2.8f}'.format(time.time() - t2))

        t1 = time.time()
        class_IDs, det_scores, det_boxes = yolo(x)
        print('Yolo Duration: {:2.8f}'.format(time.time() - t1))
        tooks['det'] += time.time() - t1

        t1 = time.time()
        person = mx.nd.array([0])
        score_threshold = mx.nd.array([0.5])
        for i, class_ID in enumerate(class_IDs[0]):
            if class_ID == person and det_scores[0][i] >= score_threshold:
                boxs.append(det_boxes[0][i].asnumpy())
        print('Filter #{} boxs: {:2.8f}'.format(len(boxs), time.time() - t1))
        tooks['det'] += time.time() - t1
        times['det'] += 1

        t1 = time.time()
        features = encoder(img, boxs)
        print('Encoder: {:2.8f}'.format(time.time() - t1))
        tooks['enc'] += time.time() - t1
        times['enc'] += 1

        t1 = time.time()
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        print('Generate Detection: {:2.8f}'.format(time.time() - t1))

        t1 = time.time()
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        print('NMS: {:2.8f}'.format(time.time() - t1))

        # Call the tracker
        t1 = time.time()
        tracker.predict()
        print('Tracker Predict: {:2.8f}'.format(time.time() - t1))

        t1 = time.time()
        tracker.update(detections)
        print('Tracker Update: {:2.8f}'.format(time.time() - t1))

        print('{} Frame: {:2.8f}\n'.format(frame_index, time.time() - t2))

        frame_index = frame_index + 1

        '''
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(img, str(track.track_id),(int(bbox[0]), int(bbox[1])+30),0, 5e-3 * 200, (0,255,0),2)

        cv2.imshow('', img)

        fps  = ( fps + (1./(time.time()-t2)) ) / 2
        print("fps= %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''

    print('Time Elapsed: {:4.8f}'.format(time.time() - t0))
    tooks['tot'] += time.time() - t0
    times['tot'] += 1

    print('Missed: {}'.format(tracker.missed))

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
