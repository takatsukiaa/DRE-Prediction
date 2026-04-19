# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'docx_guiIuITrd.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QHeaderView, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,
    QTableView, QWidget)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.parsed_table = QTableView(self.centralwidget)
        self.parsed_table.setObjectName(u"parsed_table")
        self.parsed_table.setGeometry(QRect(20, 50, 511, 411))
        self.btn_source = QPushButton(self.centralwidget)
        self.btn_source.setObjectName(u"btn_source")
        self.btn_source.setGeometry(QRect(600, 170, 151, 41))
        self.btn_output = QPushButton(self.centralwidget)
        self.btn_output.setObjectName(u"btn_output")
        self.btn_output.setGeometry(QRect(600, 260, 151, 41))
        self.data_select = QComboBox(self.centralwidget)
        self.data_select.addItem("")
        self.data_select.addItem("")
        self.data_select.addItem("")
        self.data_select.addItem("")
        self.data_select.addItem("")
        self.data_select.addItem("")
        self.data_select.setObjectName(u"data_select")
        self.data_select.setGeometry(QRect(20, 30, 69, 22))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.btn_source.setText(QCoreApplication.translate("MainWindow", u"Select Source File (Docx)", None))
        self.btn_output.setText(QCoreApplication.translate("MainWindow", u"Output to csv", None))
        self.data_select.setItemText(0, QCoreApplication.translate("MainWindow", u"Static Data", None))
        self.data_select.setItemText(1, QCoreApplication.translate("MainWindow", u"ASMs", None))
        self.data_select.setItemText(2, QCoreApplication.translate("MainWindow", u"Seizure", None))
        self.data_select.setItemText(3, QCoreApplication.translate("MainWindow", u"EEG Results", None))
        self.data_select.setItemText(4, QCoreApplication.translate("MainWindow", u"Blood Tests", None))
        self.data_select.setItemText(5, QCoreApplication.translate("MainWindow", u"Cognitive Tests", None))

    # retranslateUi

