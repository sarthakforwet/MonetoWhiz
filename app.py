from PyQt5.QtWidgets import QWidget, QApplication, QDesktopWidget, QMessageBox, QLabel, QFrame, QGridLayout
from PyQt5.QtWidgets import QPushButton, QInputDialog, QFileDialog, QLineEdit, QHBoxLayout, QSplitter, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys
from PIL import Image
import pickle
import model
import matplotlib.pyplot as plt
import cv2

class MonetoWhiz(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.lbl = None
        self.img_path = None
        self.pred = None

    def initUI(self):
        self.layout = QGridLayout(self)
        self.setWindowTitle("MonetoWhiz")
        self.center()
        self.buildArch()
        self.qButton()
        self.show()

    def buildArch(self):
        btn = QPushButton('Choose file', self)
        btn.move(100, 280)
        btn.clicked.connect(self.loadImage)
        self.layout.addWidget(btn, 2, 0)

    def loadImage(self):
        if self.lbl is not None:
            self.lbl.clear()

        file = self.openFileNameDialog()
        self.img_path = file
        pixmap = QPixmap(file)
        pixmap1 = pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.lbl = QLabel()
        self.lbl.setPixmap(pixmap1)
        self.layout.addWidget(self.lbl, 0, 0)

        # Monet button
        monet = QPushButton("Create Monet", self)
        monet.clicked.connect(self.createMonet)
        self.layout.addWidget(monet, 2, 1)

    def createMonet(self):
        pred_path = model.predict(self.img_path)
        pixmap = QPixmap(pred_path) # Try if we can achieve the same using array. It would avoid the default saving thing.
        self.lbl = QLabel()
        pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.lbl.setPixmap(pixmap)
        self.layout.addWidget(self.lbl, 0, 1)

        # We don't need it cuz now we are saving the prediction by default.
        #download = QPushButton('save monet', self)
        #self.layout.addWidget(download, 2, 2)
        #download.clicked.connect(self.downloadImage)

    def qButton(self):
        q = QPushButton("Quit", self)
        q.clicked.connect(QApplication.instance().quit)
        q.resize(q.sizeHint())
        self.layout.addWidget(q, 2, 3)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
        "All Files(*);;Image Files (*.jpg, *.png, *.jpeg)", options = options)

        return file

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.resize(500, 500)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Message", "Are you sure you wanna quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply==QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__=="__main__":
    app = QApplication(sys.argv)
    m = MonetoWhiz()
    sys.exit(app.exec_())