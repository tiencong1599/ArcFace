# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_UIT(object):
    def setupUi(self, UIT):
        UIT.setObjectName("UIT")
        UIT.resize(1322, 656)
        self.capture_icon_label = QtWidgets.QLabel(UIT)
        self.capture_icon_label.setGeometry(QtCore.QRect(10, 10, 640, 480))
        self.capture_icon_label.setObjectName("capture_icon_label")
        self.record_label = QtWidgets.QLabel(UIT)
        self.record_label.setGeometry(QtCore.QRect(660, 10, 640, 480))
        self.record_label.setObjectName("record_label")
        self.capture_btn = QtWidgets.QPushButton(UIT)
        self.capture_btn.setGeometry(QtCore.QRect(20, 530, 171, 28))
        self.capture_btn.setObjectName("capture_btn")
        self.re_train_btn = QtWidgets.QPushButton(UIT)
        self.re_train_btn.setGeometry(QtCore.QRect(350, 530, 171, 28))
        self.re_train_btn.setObjectName("re_train_btn")
        self.name_txt = QtWidgets.QLineEdit(UIT)
        self.name_txt.setGeometry(QtCore.QRect(70, 590, 211, 22))
        self.name_txt.setObjectName("name_txt")
        self.name = QtWidgets.QLabel(UIT)
        self.name.setGeometry(QtCore.QRect(20, 590, 51, 21))
        self.name.setObjectName("name")
        self.id = QtWidgets.QLabel(UIT)
        self.id.setGeometry(QtCore.QRect(20, 620, 51, 21))
        self.id.setObjectName("id")
        self.id_txt = QtWidgets.QLineEdit(UIT)
        self.id_txt.setGeometry(QtCore.QRect(70, 620, 211, 22))
        self.id_txt.setObjectName("id_txt")
        self.exit_btn = QtWidgets.QPushButton(UIT)
        self.exit_btn.setGeometry(QtCore.QRect(1130, 620, 93, 28))
        self.exit_btn.setObjectName("exit_btn")
        self.record_btn = QtWidgets.QPushButton(UIT)
        self.record_btn.setGeometry(QtCore.QRect(990, 530, 171, 28))
        self.record_btn.setObjectName("record_btn")
        self.re_embedding_btn = QtWidgets.QPushButton(UIT)
        self.re_embedding_btn.setGeometry(QtCore.QRect(670, 530, 141, 28))
        self.re_embedding_btn.setObjectName("re_embedding_btn")

        self.retranslateUi(UIT)
        QtCore.QMetaObject.connectSlotsByName(UIT)

    def retranslateUi(self, UIT):
        _translate = QtCore.QCoreApplication.translate
        UIT.setWindowTitle(_translate("UIT", "Dialog"))
        self.capture_icon_label.setText(_translate("UIT", "TextLabel"))
        self.record_label.setText(_translate("UIT", "TextLabel"))
        self.capture_btn.setText(_translate("UIT", "Capture"))
        self.re_train_btn.setText(_translate("UIT", "Re-train"))
        self.name.setText(_translate("UIT", "Name:"))
        self.id.setText(_translate("UIT", "ID:"))
        self.exit_btn.setText(_translate("UIT", "Exit"))
        self.record_btn.setText(_translate("UIT", "Record"))
        self.re_embedding_btn.setText(_translate("UIT", "Re-embedding"))