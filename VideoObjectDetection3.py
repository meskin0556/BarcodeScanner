import sys
from os import path
import datetime
import time
import threading
import cv2
import numpy as np
import tensorflow as tf

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class Worker(QtCore.QThread):
    data = QtCore.pyqtSignal(np.ndarray)
    _start = False

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self._stopped = True
        self._mutex = QtCore.QMutex()
        self._start = False
        self.vc = cv2.VideoCapture(0)
        # self.vc.set(5, 30)  #set FPS
        # self.vc.set(3, 640)  # set width
        # self.vc.set(4, 480)  # set height

        if not self.vc.isOpened():
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Failed to open camera.")
            msgBox.exec_()
            return

    def stop(self):
        self._mutex.lock()
        self._start = False
        self._stopped = True
        self._mutex.unlock()

    def run(self):
        self._stopped = False
        self.current_time = datetime.datetime.now()
        while self._start:
            print(datetime.datetime.now() - self.current_time)
            self.current_time = datetime.datetime.now()
            rval, frame = self.vc.read()

            with tf.Session(graph=detection_graph) as sess:
                with tf.Session() as sess:
                    image_np_expanded = np.expand_dims(frame, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(frame,
                                                                       np.squeeze(boxes),
                                                                       np.squeeze(classes).astype(np.int32),
                                                                       np.squeeze(scores),
                                                                       category_index,
                                                                       use_normalized_coordinates=True,
                                                                       line_thickness=8)
            self.data.emit(frame)


class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        layout = QtWidgets.QVBoxLayout()

        button_layout = QtWidgets.QHBoxLayout()

        btnCamera = QtWidgets.QPushButton("Open camera")
        btnCamera.clicked.connect(self.openCamera)
        button_layout.addWidget(btnCamera)

        btnCamera = QtWidgets.QPushButton("Stop camera")
        btnCamera.clicked.connect(self.stopCamera)
        button_layout.addWidget(btnCamera)

        layout.addLayout(button_layout)

        # Add a label
        self.label = QtWidgets.QLabel()
        self.label.setFixedSize(640, 480)
        # pixmap = self.resizeImage(filename)
        # self.label.setPixmap(pixmap)
        layout.addWidget(self.label)

        # Add a text area
        self.results = QtWidgets.QTextEdit()
        # self.readBarcode(filename)
        layout.addWidget(self.results)
        #
        # Set the layout
        self.setLayout(layout)
        self.setWindowTitle("Object Detection")
        #
        self.setFixedSize(800, 800)

        self._worker = Worker()
        # self._worker.started.connect(self.worker_started_callback)
        # self._worker.finished.connect(self.worker_finished_callback)
        self._worker.data.connect(self.worker_data_callback)

    def worker_data_callback(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)

    def openCamera(self):

        self._worker._start = True
        self._worker.start()

    def stopCamera(self):
        self._worker.stop()

    current_time = datetime.datetime.now()


def main():
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget()
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    tensorflow_filepath = path.join(script_dir,
                                'ssd_mobilenet_v1_coco_2018_01_28',
                                 'frozen_inference_graph.pb')

    tensorflow_filepath = path.abspath(tensorflow_filepath)
    label_map = label_map_util.load_labelmap('mscoco_label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(tensorflow_filepath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


main()