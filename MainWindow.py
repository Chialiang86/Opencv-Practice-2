import cv2
import sys
import numpy as np;
from scipy import signal
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QLabel, QPushButton, QGridLayout, QGroupBox, QVBoxLayout, QGridLayout


class MainWindow(QWidget):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.__cache = {}


        # 1. FindContour
        self.__btn11 = QPushButton('1.1 Draw Contour')
        self.__btn11.clicked.connect(self.__drawContour)
        self.__btn12 = QPushButton('1.2 Count Coins')
        self.__label11 = QLabel()

        # 2. Calibration
        self.__btn21 = QPushButton('2.1 Find Coners')
        self.__btn22 = QPushButton('2.2 Find Intrinsic')
        # 2.3 Find Extrinsic
        self.__label23 = QLabel('Select image')
        self.__comb23 = QComboBox()
        self.__btn23 = QPushButton('2.3 Find Extrinsic')
        self.__btn24 = QPushButton('2.4 Find Distortion')

        # 3. Augmented Reality
        self.__btn31 = QPushButton('3.1 Augmented Reality')

        # 4. Stereo Disparity Map
        self.__btn41 = QPushButton('4.1 Stereo Disparty Map')

        self.__set_UIlayout()
        self.show()
        self.resize(600, 500)
        self.setWindowTitle('Opencv 2020 HW2')
        

    def __set_UIlayout(self):
        window_grid = QGridLayout();

        gb1_grid = QGridLayout();
        gb1 = QGroupBox('1. Find Contour')
        gb1_grid.addWidget(self.__btn11, 1, 1, 2, 1)
        gb1_grid.addWidget(self.__btn12, 3, 1, 2, 1)
        gb1_grid.addWidget(self.__label11, 5, 1, 2, 1)
        gb1.setLayout(gb1_grid)
        window_grid.addWidget(gb1, 1, 1, 2, 2)

        gb2_grid = QGridLayout();
        gb2 = QGroupBox('2. Calibration')
        gb2_grid.addWidget(self.__btn21, 1, 1, 1, 1)
        gb2_grid.addWidget(self.__btn22, 2, 1, 1, 1)
        # 2.3 widgets
        gb23_grid = QGridLayout();
        gb23 = QGroupBox('2.3 Find Extrinsic')
        gb23_grid.addWidget(self.__label23, 1, 1, 1, 1)
        gb23_grid.addWidget(self.__comb23, 2, 1, 1, 1)
        gb23_grid.addWidget(self.__btn23, 3, 1, 1, 1)
        gb23.setLayout(gb23_grid)
        gb2_grid.addWidget(gb23, 1, 3, 3, 1)
        gb2_grid.addWidget(self.__btn24, 3, 1, 1, 1)
        gb2.setLayout(gb2_grid)
        window_grid.addWidget(gb2, 3, 1, 2, 4)

        gb3_grid = QGridLayout();
        gb3 = QGroupBox('3. Augmented Reality')
        gb3_grid.addWidget(self.__btn31)
        gb3.setLayout(gb3_grid)
        window_grid.addWidget(gb3, 1, 3, 1, 2)
    
        gb4_grid = QGridLayout();
        gb4 = QGroupBox('4. Stereo Disparity Map')
        gb4_grid.addWidget(self.__btn41)
        gb4.setLayout(gb4_grid)
        window_grid.addWidget(gb4, 2, 3, 1, 2)

        self.setLayout(window_grid)

    def __drawContour(self):
        img1 = cv2.imread('Datasets/Q1_Image/coin01.jpg')
        img2 = cv2.imread('Datasets/Q1_Image/coin02.jpg')
        contours1, img1 = self.__binartAndCountours(img1)
        contours2, img2 = self.__binartAndCountours(img2)
        self.__cache['1.1-1'] = contours1
        self.__cache['1.1-2'] = contours2
        cv2.imshow("1.1-1 contour", img1)
        cv2.imshow("1.1-2 contour", img2)
        cv2.waitKey(0)

    def __setCountours(self, src):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.Canny(gray, 100, 300)
        ret, binary = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(src, contours, -1, (255, 255, 0), 1)
        return contours, src


        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec_())


