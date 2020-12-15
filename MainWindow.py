import cv2
import sys
import numpy as np;
from scipy import signal
from PyQt5.QtWidgets import QFileDialog ,QApplication, QWidget, QComboBox, QLabel, QPushButton, QGridLayout, QGroupBox, QVBoxLayout, QGridLayout, QMessageBox


class MainWindow(QWidget):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.__cache = {}

        self.__msgBox =  QMessageBox()
        self.__msgBox.setIcon(QMessageBox.Information)

        # 1. FindContour
        self.__btn11 = QPushButton('1.1 Draw Contour')
        self.__btn11.clicked.connect(self.drawContour11)
        self.__btn12 = QPushButton('1.2 Count Coins')
        self.__btn12.clicked.connect(self.countCoins12)
        self.__label12 = QLabel()
        self.__label12.setText('There are _ coins')

        # 2. Calibration
        self.__btn21 = QPushButton('2.1 Find Coners')
        self.__btn21.clicked.connect(self.findCorner21)
        self.__btn22 = QPushButton('2.2 Find Intrinsic')
        self.__btn22.clicked.connect(self.findIntrinsic22)
        # 2.3 Find Extrinsic
        self.__label23 = QLabel('Select image')
        self.__comb23 = QComboBox()
        for i in range(15):
            self.__comb23.addItem(str(i + 1))
        self.__index23()
        self.__comb23.currentIndexChanged.connect(self.__index23)
        self.__btn23 = QPushButton('2.3 Find Extrinsic')
        self.__btn23.clicked.connect(self.findExtrinsic23)
        self.__btn24 = QPushButton('2.4 Find Distortion')
        self.__btn24.clicked.connect(self.findDistortion24)

        # 3. Augmented Reality
        self.__btn31 = QPushButton('3.1 Augmented Reality')

        # 4. Stereo Disparity Map
        self.__btn41 = QPushButton('4.1 Stereo Disparty Map')

        self.__set_UIlayout()
        self.show()
        self.resize(600, 500)
        self.setWindowTitle('Opencv 2020 HW2')
        

    def __set_UIlayout(self):
        window_grid = QGridLayout()

        gb1_grid = QGridLayout()
        gb1 = QGroupBox('1. Find Contour')
        gb1_grid.addWidget(self.__btn11, 1, 1, 2, 1)
        gb1_grid.addWidget(self.__btn12, 3, 1, 2, 1)
        gb1_grid.addWidget(self.__label12, 5, 1, 2, 1)
        gb1.setLayout(gb1_grid)
        window_grid.addWidget(gb1, 1, 1, 2, 2)

        gb2_grid = QGridLayout()
        gb2 = QGroupBox('2. Calibration')
        gb2_grid.addWidget(self.__btn21, 1, 1, 1, 1)
        gb2_grid.addWidget(self.__btn22, 2, 1, 1, 1)
        # 2.3 widgets
        gb23_grid = QGridLayout()
        gb23 = QGroupBox('2.3 Find Extrinsic')
        gb23_grid.addWidget(self.__label23, 1, 1, 1, 1)
        gb23_grid.addWidget(self.__comb23, 2, 1, 1, 1)
        gb23_grid.addWidget(self.__btn23, 3, 1, 1, 1)
        gb23.setLayout(gb23_grid)
        gb2_grid.addWidget(gb23, 1, 3, 3, 1)
        gb2_grid.addWidget(self.__btn24, 3, 1, 1, 1)
        gb2.setLayout(gb2_grid)
        window_grid.addWidget(gb2, 3, 1, 2, 4)

        gb3_grid = QGridLayout()
        gb3 = QGroupBox('3. Augmented Reality')
        gb3_grid.addWidget(self.__btn31)
        gb3.setLayout(gb3_grid)
        window_grid.addWidget(gb3, 1, 3, 1, 2)
    
        gb4_grid = QGridLayout()
        gb4 = QGroupBox('4. Stereo Disparity Map')
        gb4_grid.addWidget(self.__btn41)
        gb4.setLayout(gb4_grid)
        window_grid.addWidget(gb4, 2, 3, 1, 2)

        self.setLayout(window_grid)

    def drawContour11(self):
        img1 = cv2.imread('Datasets/Q1_Image/coin01.jpg')
        img2 = cv2.imread('Datasets/Q1_Image/coin02.jpg')
        contours1, img1 = self.__setContours(img1)
        contours2, img2 = self.__setContours(img2)
        self.__cache['1.1-1'] = len(contours1)
        self.__cache['1.1-2'] = len(contours2)
        cv2.startWindowThread()
        cv2.imshow("1.1-1 contour", img1)
        cv2.imshow("1.1-2 contour", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __setContours(self, src):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.Canny(gray, 100, 250)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.RETR_EXTERNA是檢測外輪廓
        cv2.drawContours(src, contours, -1, (255, 255, 0), 1)
        return contours, src

    def countCoins12(self):
        msg = 'There are {cnt1} coins in coin01.jpg\nThere are {cnt2} coins in coin01.jpg'.format(cnt1 = self.__cache['1.1-1'], cnt2 = self.__cache['1.1-2'])
        self.__label12.setText(msg)
        self.__label12.update()

    def __index23(self):
        self.__cache['index23'] = self.__comb23.currentIndex()
        fname = 'Datasets/Q2_Image/' + self.__comb23.currentText() + '.bmp'
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corner = cv2.findChessboardCorners(gray, (11, 8),flags=cv2.CALIB_CB_FAST_CHECK)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if(found):
            cv2.cornerSubPix(gray, corner, (5,5), (-1,1), criteria)
        self.__cache['2-fname'] = fname
        self.__cache['2-img'] = img
        self.__cache['2-gray'] = gray
        self.__cache['2-bSize'] = (11, 8)
        self.__cache['2-criteria'] = criteria
        self.__cache['2-corner'] = corner
        self.__cache['2-found'] = found

    def findCorner21(self):
        img = self.__cache['2-img']
        cv2.drawChessboardCorners(img, (11, 8), self.__cache['2-corner'], self.__cache['2-found'])
        cv2.imshow('Corner', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findIntrinsic22(self):
        c ,r = 11, 8
        objP3D = np.zeros((1, c * r, 3), np.float32)
        objP3D[0, :, :2] = np.mgrid[0:c, 0:r].T.reshape(-1, 2)
        objP2D = cv2.cornerSubPix(self.__cache['2-gray'], self.__cache['2-corner'], (11, 11), (-1, -1), self.__cache['2-criteria'])
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera([objP3D], [objP2D], self.__cache['2-gray'].shape[::-1], None, None)
        self.__cache['2-distortion'] = distortion
        print('distortion')
        print(distortion)
        print(self.__cache['2-fname'])
        print(matrix)

    def findExtrinsic23(self):
        c, r = 11, 8
        objP3D = np.zeros((1, c * r, 3), np.float32)
        objP3D[0, :, :2] = np.mgrid[0:c, 0:r].T.reshape(-1, 2)
        objP2D = cv2.cornerSubPix(self.__cache['2-gray'], self.__cache['2-corner'], (11, 11), (-1, -1), self.__cache['2-criteria'])
        ret, A, distortion, rvec, tvec = cv2.calibrateCamera([objP3D], [objP2D], self.__cache['2-gray'].shape[::-1], None, None)
        r, j = cv2.Rodrigues(np.float32(rvec))
        t = [tvec[0][0], tvec[0][1], tvec[0][2]]
        res = np.hstack((r, t))
        self.__cache['2-distortion'] = distortion
        print(self.__cache['2-fname'])
        print(res)

    def findDistortion24(self):
        if '2-distortion' in self.__cache:
            print(self.__cache['2-distortion'])
        else :
            self.__msgBox.setText('2-2 or 2-3 not done yet')
            self.__msgBox.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec_())


