import sys
import cv2
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from utillc import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, \
    QLabel, QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox
from PyQt5.QtCore import Qt, QEvent, QSize, QObject, QTimer, \
    QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtCore, QtGui
import sys

from matplotlib import pyplot as plt


import torch
from torch import nn


import numpy as np

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.timeToStartNextAnimation = 10
        self.title = "Image Viewer"
        self.setWindowTitle(self.title)
        self.animationTimer = QTimer()
        self.animationTimer.timeout.connect(self.step)
        self.label = QLabel(self)
        pixmap = QPixmap('shinchan.png')
        self.label.setPixmap(pixmap)
        self.setCentralWidget(self.label)
        self.resize(pixmap.width(), pixmap.height())
        self.count = 0

        self.image = cv2.imread('shinchan.png')[:,:,0]
        self.black = self.image * 0
        self.white  = self.black + 255
        self.back = np.where(self.image > 200, self.image, self.white)[:,:]

        #self.back = self.white
        #self.back = self.white

        self.im = self.white
        self.im[0,0] = 1

        self.im = np.where(self.image > 122, self.white, self.black)

        self.im = self.black
        self.im[0,0] = 255

        self.x = torch.tensor(self.im)[None, ...].float() / 255
        self.zeros = torch.zeros_like(self.x)
        self.ones = torch.ones_like(self.x)
        EKOX(torch.mean(self.x))
        
        nb_channels = 1
        h, w = 5, 5
        x = torch.randn(1, nb_channels, h, w)
        weights = torch.tensor(np.ones(shape=(3,3))).float()
        weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)       
        self.conv = nn.Conv2d(nb_channels, 1, 3, bias=False, padding='same')
        with torch.no_grad():
            self.conv.weight = nn.Parameter(weights)
        
        
    def step(self) -> None :
        self.count += 1;
        with torch.no_grad():
            output = self.conv(self.x)
            #EKON(output.shape, self.zeros.shape, self.ones.shape)
            self.x = torch.clamp(output, self.zeros, self.ones)
        #EKOX(output.shape)
        imm = self.im.copy()
        #EKOX(torch.mean(self.x))
        self.im = imm = self.x.detach().byte().numpy()[0, ...] * 255
        #EKOX(np.mean(imm))
        #plt.imshow(imm); plt.show()
        
        #EKO();
        h,w = imm.shape
        bytesPerLine = 1 * w
        #EKOX(self.image)
        #self.image += 1
        i = QImage(imm, w, h, w, QImage.Format_Grayscale8)
        self.label.setPixmap(QPixmap(i))
    

    def start(self) :
        self.animationTimer.start(self.timeToStartNextAnimation)

EKO()
app = QApplication(sys.argv)
w = MainWindow()
w.show()
w.start()
EKO()
sys.exit(app.exec_())


