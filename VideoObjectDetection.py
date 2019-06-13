import sys
from os import path

import cv2
import numpy as np
import tensorflow as tf

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=1, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)


class ObjectDetectionWidget(QtWidgets.QWidget):
    def __init__(self, tensorflow_filepath, parent=None):
        super().__init__(parent)
        with tf.gfile.FastGFile(tensorflow_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Session() as sess:
            # Restore session
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    def image_data_slot(self, image_data):
        with tf.Session() as sess:
            objects = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                sess.graph.get_tensor_by_name('detection_classes:0')],
                               feed_dict={'image_tensor:0': image_data.reshape(1, image_data.shape[0], image_data.shape[1], 3)})
        num_detections = int(objects[0][0])
        for i in range(num_detections):
            classId = int(objects[3][0][i])
            score = float(objects[1][0][i])
            bbox = [float(v) for v in objects[2][0][i]]
            rows = image_data.shape[0]
            cols = image_data.shape[1]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv2.rectangle(image_data, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())
        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, tensorflow_filepath, parent=None):
        super().__init__(parent)
        fp = tensorflow_filepath
        self.object_detection_widget = ObjectDetectionWidget(fp)

        # TODO: set video port
        self.record_video = RecordVideo()

        image_data_slot = self.object_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.object_detection_widget)

        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)

        self.run_button.clicked.connect(self.record_video.start_recording)

        # self.dection_button = QtWidgets.QPushButton('detection')
        # layout.addWidget(self.dection_button)
        #
        # self.dection_button.clicked.connect(self.object_detection_widget.obejct_detection_slot)
        self.setLayout(layout)


def main(tensorflow_filepath):
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(tensorflow_filepath)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    tensorflow_filepath = path.join(script_dir,
                                'ssd_mobilenet_v1_coco_2018_01_28',
                                 'frozen_inference_graph.pb')

    tensorflow_filepath = path.abspath(tensorflow_filepath)

main(tensorflow_filepath)