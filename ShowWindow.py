import cv2
import numpy as np;
from PyQt5.QtWidgets import QFileDialog ,QApplication, QWidget, QComboBox, QLabel, QPushButton, QGridLayout, QGroupBox, QVBoxLayout, QGridLayout, QMessageBox
from PyQt5.QtGui import QPixmap
import imageio
from matplotlib import pyplot as plt

class ShowWindow(QWidget):
    def __init__(self, img = None):
        super(self.__class__, self).__init__()
        
        self.resize(img.shape[0], img.shape[1])

        self.__img = QPixmap(img)

        self.__label = QLabel()
        self.__label.setGeometry(0, 0, img.shape[0], img.shape[1])
        self.__label.setPixmap(self, self.__img)
        self.__label.show()

        self.__info = QLabel()
        self.__info.setGeometry(img.shape[0] - 100, img.shape[1] - 50, 100, 50)
        self.__info.setText('demo')
        self.__info.show()

        self.show()

    def setImg(self, img):
        print
