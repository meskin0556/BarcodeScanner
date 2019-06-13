import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from os import path
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt


import cv2

cap = cv2.VideoCapture(1)




script_dir = path.dirname(path.realpath(__file__))
tensorflow_filepath = path.join(script_dir,
                                'ssd_mobilenet_v1_coco_2018_01_28',
                                'frozen_inference_graph.pb')

tensorflow_filepath = path.abspath(tensorflow_filepath)



detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(tensorflow_filepath, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            # boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            # scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            objects = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                sess.graph.get_tensor_by_name('detection_classes:0')],
                               feed_dict={
                                   'image_tensor:0': image_np.reshape(1, image_np.shape[0], image_np.shape[1],
                                                                        3)})
            num_detections = int(objects[0][0])
            # Visualization of the results of a detection.
            objectNumber = int(objects[0][0])
            for i in range(objectNumber):
                classId = int(objects[3][0][i])
                score = float(objects[1][0][i])
                bbox = [float(v) for v in objects[2][0][i]]
                rows = image_np.shape[0]
                cols = image_np.shape[1]
                if score > 0.3:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv2.rectangle(image_np, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break