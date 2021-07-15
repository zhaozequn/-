# -*- coding: utf-8 -*-
import os
import sys
import threading
import time
from threading import Thread

import cv2
import numpy as np
import paddle
import paddle.fluid as fluid
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import gxipy as gx
from GUI615 import Ui_Form
from ThreadWithReturn import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

paddle.enable_static()


def White_balance(img):
    b, g, r = cv2.split(img)  # 将原图分为三通道

    r_avg = cv2.mean(r)[0]  # 求出每一个通道均值
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    k = (r_avg + g_avg + b_avg) / 3

    kr = k / r_avg  # 求每个通道的增益
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

    frameWhite_balabce = cv2.merge([b, g, r])  # 三通道数值合成

    return frameWhite_balabce


def Morphological(img):
    (b, g, r) = cv2.split(img)  # 得到bgr通道图像，采用b通道
    # 对b通道图像进行形态学处理得到Mask
    # gray = cv2.cvtColor(frameProcess,cv2.COLOR_BGR2GRAY)#转化为灰度图
    # gray_diff = cv2.absdiff(gray,previous)#与上一帧计算绝对值差
    gray = cv2.medianBlur(b, 3)  # 中值滤波
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 5*5内核高斯滤波
    ret, mask = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY)  # 阈值分割，黑白对调

    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)  # 腐蚀算法
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)  # 膨胀算法

    return mask


def Roi_area(img_in):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_in,
                                                                            connectivity=8)  # 在mask中找到所需要的所有轮廓及信息
    area_totall = 0
    for i in range(1, num_labels):  # 在循环中找到最大的连通域
        area = stats[i, 4]
        # if area >= 50000:
        # max_area[j] = area
        area_totall += area
    print(area_totall)
    return area_totall


def Roi_frame(yuantu, img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)  # 在mask中找到所需要的所有轮廓及信息
    max_area = 0
    j = []  # 统计符合大小的个数
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):  # 在循环中找到最大的连通域
        area = stats[i, 4]
        if 20000 < area < 100000:
            # max_area[j] = area
            j.append(i)
    for i in range(0, len(j)):
        img = labels == j[i]  # ==优先级更高，先比较labels如果等于j，则img=1，否则为0
        output[img] = 255
    roi = cv2.bitwise_and(yuantu, output)  # 将白区域与原图与操作
    output, output, output = cv2.split(output)
    return roi, output


# 抓取合适的轮廓并计数
def junde_and_count(i, frame, roi, frame_num, roi_deeplearn):  # i表示的是一帧中有多少个轮廓被选取
    global c
    global CapArea
    result = 0
    # roi_deeplearn = np.zeros((492, 656, 3), dtype='uint8')
    if 20000 < cv2.contourArea(i) < 100000:
        M = cv2.moments(i)  # 找图像矩
        if M["m00"] != 0:  # 图像面积为零
            cX = int(M["m10"] / M["m00"])  # 找到轮廓中心的X坐标
            cY = int(M["m01"] / M["m00"])  # 找到轮廓中心的Y坐标

            if frame_num % 50 == 0:  # 到中间范围计数，这里要改，实现一一计数
                cv2.circle(frame, (cX, cY), 12, (70, 70, 70), -1)  # 中心点画圆
                cv2.imwrite("roi.bmp", roi)
                roi_deeplearn = roi
                print('junde_and_count工作，当前线程计数', threading.enumerate())  # 计线程数
                # -------------------------------------------------------------------------------------------------------------
                # th_deeplearning = DeepLearning()
                # result = DeepLearning().deeplearning_predict()
                # th_deeplearning.start()

            cv2.drawContours(frame, i, -1, (255, 70, 70), 2)  # 画出轮廓
            area = cv2.contourArea(i)  # 被capture的计算面积

            if area > 0:  # 计算面积
                CapArea = area

        # cv2.putText(roi,"total num: " +str(count_num) , (0,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

    return CapArea, frame_num, frame, roi, roi_deeplearn, result


class Gui(QMainWindow):

    def __init__(self):
        super(Gui, self).__init__()  # super构造器返回父级对象
        self.initUI()

    def initUI(self):
        self.statusbar = self.statusBar()
        # 子菜单栏
        exitAct = QAction(QIcon('quit.jpg'), 'Exit', self)
        exitAct.setShortcut('&Exit')
        exitAct.triggered.connect(quit)

        # menubar就是父菜单栏
        menubar = self.menuBar()
        fileMenu_file = menubar.addMenu('&File')
        fileMenu_setting = menubar.addMenu('&setting')
        fileMenu_file.addAction(exitAct)  # 添加exitAct到filemenu下

        # self.setGeometry(500,200,1000,700)
        self.resize(1000, 700)
        self.center()
        self.setWindowTitle("苹果分级系统")
        self.setWindowIcon(QIcon('icon.jpg'))
        self.statusBar().showMessage("Ready")

        btn1 = QPushButton('连接相机', self)
        btn1.setToolTip("连接相机1,2")
        btn1.clicked.connect(self.buttonClicked)  # 在status显示
        btn1.resize(100, 20)
        btn1.move(50, 50)

        btn2 = QPushButton('显示图像', self)
        btn2.setToolTip('显示图像')
        btn2.clicked.connect(self.picture)
        btn2.resize(100, 20)
        btn2.move(200, 50)

        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def buttonClicked(self):
        sender = self.sender()
        self.statusBar().showMessage(sender.text())

    def picture(self):
        self.buttonClicked()
        self.img = cv2.imread('file.jpg')
        QPicture

    def keyPressEvent(self, e):  # e就是重写了新的方法
        if e.key() == Qt.Key_Escape:
            self.close()


class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None



class Gui615(QMainWindow, Ui_Form):
    def __init__(self):
        super(Gui615, self).__init__()
        self.setupUi(self)
        self.btn_connect_device.clicked.connect(self.connectPLC)
        self.btn_open_pic.clicked.connect(self.OpenPicture)
        self.btn_open_camera.clicked.connect(self.OpenCamera)
        self.btn_close_camera.clicked.connect(self.closecam)
        self.close1 = False
        self.starttime = time.time()

    def closecam(self):
        self.close1 = True

    def connectPLC(self):
        pass

    def OpenPicture(self):
        print(threading.enumerate())
        list_name = os.listdir("F:\\apple splice\\145/")
        list_name.sort(key=lambda x: int(x[:-4]))  # 排序
        for i in list_name:
            img = cv2.imread("F:\\apple splice\\145/" + i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)  # 将图像转化为Qimage
            img = QPixmap(img).scaled(self.label_Picture_1capture.width(), self.label_Picture_1capture.height())
            self.label_Picture_1capture.setPixmap(img)
            QApplication.processEvents()  # 刷新界面显示
            time.sleep(0.05)

    def OpenCamera(self):
        # 在新线程中开启采集图像
        # thread_show = threading.Thread(target=self.thread_show)
        # th1 = ThreadShow()
        # thread_show.start()
        self.thread_show()
        length = len(threading.enumerate())
        print('线程叔叔', length)

    def While_Process_Doublecam(self, cam1, cam2, frame_num, roi_deeplearn1, roi_deeplearn2):
        # get raw image
        raw_image1 = cam1.data_stream[0].get_image()
        raw_image2 = cam2.data_stream[0].get_image()

        if raw_image1 is None:
            print("Getting image failed.")
        if raw_image2 is None:
            print("Getting image failed.")

        # get RGB image from raw image
        rgb_image1 = raw_image1.convert("RGB")
        rgb_image2 = raw_image2.convert("RGB")

        # improve image quality
        # rgb_image1.image_improvement(color_correction_param, contrast_lut, gamma_lut)
        # rgb_image2.image_improvement(color_correction_param, contrast_lut, gamma_lut)

        # create numpy array with data from raw image
        numpy_image1 = rgb_image1.get_numpy_array()
        numpy_image2 = rgb_image2.get_numpy_array()

        predictresult1 = None
        predictresult2 = None

        red_ratio1 = 0
        red_ratio2 = 0

        # -----------------------------------相  机   1  ----------------------------------------
        frame_yuan1 = White_balance(numpy_image1)  # -------至此，得到相机1的np图像,frame1_yuan是3通道的原图
        frame1 = frame_yuan1
        mask1 = Morphological(frame1)  # 先得到一个mask，因为是单维度的，不能用于显示
        roi1, output1 = Roi_frame(frame_yuan1, mask1)  # 用于在显示ROI时消除小空洞,注意此处有区域面积限制,output与roi相同，
        # 只是一张单通道黑白图,output用来计算面积
        contours1, hierarchy = cv2.findContours(mask1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in contours1:
            CapArea1, frame_num, frame1, roi1, roi_deeplearn1, predictresult1 = junde_and_count(i, frame1,
                                                                                                roi1,
                                                                                                frame_num,
                                                                                                roi_deeplearn1)  # 得到面积caparea但暂时无用，面积
        #----------------这里得到roi_deeplearn_show
        roi_deeplearn_show1 = QImage(roi_deeplearn1[:], roi_deeplearn1.shape[1], roi_deeplearn1.shape[0],
                                     roi_deeplearn1.shape[1] * 3,
                                     QImage.Format_RGB888)  # 将图像转化为Qimage
        roi_deeplearn_show1 = QPixmap(roi_deeplearn_show1).scaled(self.label_Picture_1capture.width(),
                                                                  self.label_Picture_1capture.height())  # 合适大小

        # 由后续Roi_Area计算
        area_all1 = Roi_area(output1)
        # 对roi进行处理，得到绿色通道图像来计算绿色区域面积
        roi1_b, roi1_g, roi1_r = cv2.split(roi1)
        ret2, green_mask1 = cv2.threshold(roi1_g, 110, 255, cv2.THRESH_BINARY)  # 分出绿色区域
        area_green1 = Roi_area(green_mask1)

        # 计算红色比例
        if area_all1 == 0:
            self.label_redratio_1.setText("红色比例：无")
        else:
            red_ratio1 = (area_all1 - area_green1) / area_all1 * 100
            red_ratio1 = round(red_ratio1, 1)

        # 对图像进行显示
        #-------------这里得到img1
        frame1 = QImage(frame1[:], frame1.shape[1], frame1.shape[0], frame1.shape[1] * 3,
                        QImage.Format_RGB888)  # 将图像转化为Qimage
        img1 = QPixmap(frame1).scaled(self.label_Picture_1.width(), self.label_Picture_1.height())  # 合适大小

        #-------------这里得到roi1
        roi1 = QImage(roi1[:], roi1.shape[1], roi1.shape[0], roi1.shape[1] * 3,
                      QImage.Format_RGB888)  # 将图像转化为Qimage
        roi1 = QPixmap(roi1).scaled(self.label_Picture_1process.width(),
                                    self.label_Picture_1process.height())  # 合适大小

        # -----------------------------------相  机   2  ----------------------------------------
        frame2_yuan = White_balance(numpy_image2)  # -------至此，得到相机2的np图像
        frame2 = frame2_yuan
        mask2 = Morphological(frame2)
        roi2, output2 = Roi_frame(frame2_yuan, mask2)  # 用于在显示ROI时消除小空洞,注意此处有区域面积限制,output与roi相同，
        # 只是一张单通道黑白图,output用来计算面积
        contours2, hierarchy = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        frame_num += 1
        for i in contours2:
            CapArea2, frame_num, frame2, roi1, roi_deeplearn2, predictresult2 = junde_and_count(i, frame2,
                                                                                                roi2,
                                                                                                frame_num,
                                                                                                roi_deeplearn2)  # 得到面积caparea但暂时无用，面积

        roi_deeplearn_show2 = QImage(roi_deeplearn2[:], roi_deeplearn2.shape[1], roi_deeplearn2.shape[0],
                                     roi_deeplearn2.shape[1] * 3,
                                     QImage.Format_RGB888)  # 将图像转化为Qimage
        roi_deeplearn_show2 = QPixmap(roi_deeplearn_show2).scaled(self.label_Picture_2capture.width(),
                                                                  self.label_Picture_2capture.height())  # 合适大小

        # 由后续Roi_Area计算
        area_all2 = Roi_area(output2)
        # 对roi进行处理，得到绿色通道图像来计算绿色区域面积
        roi2_b, roi2_g, roi2_r = cv2.split(roi2)
        ret2, green_mask2 = cv2.threshold(roi2_g, 110, 255, cv2.THRESH_BINARY)  # 分出绿色区域
        area_green2 = Roi_area(green_mask2)

        # 计算红色比例
        if area_all2 == 0:
            self.label_redratio_2.setText("红色比例：无")
        else:
            red_ratio2 = (area_all2 - area_green2) / area_all2 * 100
            red_ratio2 = round(red_ratio2, 1)

        # ---------------------以上是处理，以下是显示-----------------

        frame2 = QImage(frame2[:], frame2.shape[1], frame2.shape[0], frame2.shape[1] * 3,
                        QImage.Format_RGB888)  # 将图像转化为Qimage
        img2 = QPixmap(frame2).scaled(self.label_Picture_2.width(), self.label_Picture_2.height())

        roi2 = QImage(roi2[:], roi2.shape[1], roi2.shape[0], roi2.shape[1] * 3,
                      QImage.Format_RGB888)  # 将图像转化为QImage
        roi2 = QPixmap(roi2).scaled(self.label_Picture_2process.width(),
                                    self.label_Picture_2process.height())  # 合适大小

        """self.endtime = time.time()
        self.runtime = round((self.endtime - self.starttime), 2)
        self.label_time.setText('运行时间：' + str(self.runtime) + 's')"""

        return predictresult1, predictresult2, area_all1, area_all2, red_ratio1, red_ratio2, img1, img2, roi_deeplearn_show1, roi_deeplearn_show2, roi1, roi2, frame_num


    def While_Process_Singlecam(self,cam,frame_num,roi_deeplearn):
        # get raw image
        raw_image = cam.data_stream[0].get_image()

        if raw_image is None:
            print("Getting image failed.")

        # get RGB image from raw image
        rgb_image = raw_image.convert("RGB")

        # create numpy array with data from raw image
        numpy_image = rgb_image.get_numpy_array()

        predictresult = None
        red_ratio = 0
        # 相机
        frame_yuan = White_balance(numpy_image)
        frame = frame_yuan
        mask = Morphological(frame)
        roi, output = Roi_frame(frame_yuan, mask)  # 用于在显示ROI时消除小空洞,注意此处有区域面积限制,output与roi相同，
        # 只是一张单通道黑白图,output用来计算面积
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        frame_num += 1

        for i in contours:
            CapArea, frame_num, frame, roi, roi_deeplearn, predictresult = junde_and_count(i, frame, roi,
                                                                                           frame_num,
                                                                                           roi_deeplearn)  # 得到面积caparea但暂时无用，面积


        # 这里 roi_deeplearn_show是为了避免roi_deeplearn不能刷新导致后续一直是Qimage格式
        roi_deeplearn_show = QImage(roi_deeplearn[:], roi_deeplearn.shape[1], roi_deeplearn.shape[0],
                                    roi_deeplearn.shape[1] * 3,
                                    QImage.Format_RGB888)  # 将图像转化为Qimage
        roi_deeplearn_show = QPixmap(roi_deeplearn_show).scaled(self.label_Picture_1capture.width(),
                                                                self.label_Picture_1capture.height())  # 合适大小

        area_all = Roi_area(output)

        # 对roi进行处理，得到绿色通道图像来计算绿色区域面积
        roi_b, roi_g, roi_r = cv2.split(roi)
        ret2, green_mask = cv2.threshold(roi_g, 110, 255, cv2.THRESH_BINARY)  # 分出绿色区域
        area_green = Roi_area(green_mask)

        # 计算红色比例
        if area_all == 0:
            self.label_redratio_1.setText("红色比例：无")
        else:
            red_ratio = (area_all - area_green) / area_all * 100
            red_ratio = round(red_ratio, 1)


        """self.endtime = time.time()
        self.runtime = round((self.endtime - self.starttime), 2)
        self.label_time.setText('运行时间：' + str(self.runtime) + 's')"""


        # -----------------------------------以上是处理--------------------以下是转化显示---------------------------
        frame = QImage(frame[:], frame.shape[1], frame.shape[0], frame.shape[1] * 3,
                       QImage.Format_RGB888)  # 将图像转化为Qimage
        img = QPixmap(frame).scaled(self.label_Picture_1.width(), self.label_Picture_1.height())

        roi = QImage(roi[:], roi.shape[1], roi.shape[0], roi.shape[1] * 3,
                     QImage.Format_RGB888)  # 将图像转化为QImage
        roi = QPixmap(roi).scaled(self.label_Picture_1process.width(),
                                  self.label_Picture_1process.height())  # 合适大小



        return predictresult,red_ratio,img,roi_deeplearn_show,roi,frame_num


    def thread_show(self):
        length = len(threading.enumerate())
        print('线程数', length)
        frame_num = 0  # 帧计数，用来控制多少次进入一次判断
        roi_deeplearn1 = np.zeros((492, 656, 3), dtype='uint8')  # 第三张图的填充背景
        roi_deeplearn2 = np.zeros((492, 656, 3), dtype='uint8')  # 第三张图的填充背景

        if self.double_camera.isChecked() == True:  # 如果双相机被选中
            device_manager = gx.DeviceManager()
            dev_num, dev_info_list = device_manager.update_device_list()
            if dev_num == 0:
                print("Number of enumerated devices is 0")
                self.showmessage_error()

            # open the first device
            cam1 = device_manager.open_device_by_index(1)
            cam2 = device_manager.open_device_by_index(2)

            # set continuous acquisition
            cam1.TriggerMode.set(gx.GxSwitchEntry.OFF)
            cam2.TriggerMode.set(gx.GxSwitchEntry.OFF)
            # set exposure
            cam1.ExposureTime.set(10000.0)
            cam2.ExposureTime.set(10000.0)
            # set gain
            cam1.Gain.set(100.0)
            cam2.Gain.set(100.0)

            cam1.stream_on()
            cam2.stream_on()

            while 1:
                while_process_doublecam = ThreadWithReturnValue(target=self.While_Process_Doublecam, args=(
                cam1, cam2, frame_num, roi_deeplearn1, roi_deeplearn2))  # 开启带参数的线程
                while_process_doublecam.start()
                predictresult1, predictresult2, area_all1, area_all2, red_ratio1, red_ratio2, img1, img2, roi_deeplearn_show1, \
                roi_deeplearn_show2, roi1, roi2, frame_num = while_process_doublecam.join()

                if predictresult1 != None:
                    self.label_defectkinds_1.setText("预测种类：" + str(predictresult1))
                if predictresult2 != None:
                    self.label_defectkinds_2.setText("预测种类：" + str(predictresult2))

                self.label_redratio_1.setText("红色比例：" + str(red_ratio1) + "%")
                self.label_redratio_2.setText("红色比例：" + str(red_ratio2) + "%")

                self.label_Picture_1.setPixmap(img1)  # 图像显示在label上
                self.label_Picture_2.setPixmap(img2)  # 图像显示在label

                self.label_Picture_1capture.setPixmap(roi_deeplearn_show1)  # 图像显示在label上
                self.label_Picture_2capture.setPixmap(roi_deeplearn_show2)  # 图像显示在label上

                self.label_Picture_1process.setPixmap(roi1)  # 图像显示在label上
                self.label_Picture_2process.setPixmap(roi2)

                self.label_framenum.setText("当前帧：" + str(frame_num))

                if self.close1:  # 判断关闭按钮是否被按下

                    blackimg = np.zeros((492, 656, 3), dtype='uint8')
                    blackimg = QImage(blackimg[:], blackimg.shape[1], blackimg.shape[0], blackimg.shape[1] * 3,
                                      QImage.Format_RGB888)
                    blackimg = QPixmap(blackimg).scaled(self.label_Picture_1.width(), self.label_Picture_1.height())
                    self.label_Picture_1.setPixmap(blackimg)
                    self.label_Picture_2.setPixmap(blackimg)
                    self.label_Picture_1process.setPixmap(blackimg)
                    self.label_Picture_2process.setPixmap(blackimg)
                    self.close1 = False
                    break


        else:
            device_manager = gx.DeviceManager()
            dev_num, dev_info_list = device_manager.update_device_list()
            if dev_num == 0:
                print("Number of enumerated devices is 0")
                self.showmessage_error()  #

            # open the first device
            cam = device_manager.open_device_by_index(1)

            # set continuous acquisition
            cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
            # set exposure
            cam.ExposureTime.set(10000.0)
            # set gain
            cam.Gain.set(100.0)
            # get stream
            cam.stream_on()

            # 设置一张纯黑的图用在第三张图上
            roi_deeplearn = np.zeros((492, 656, 3), dtype='uint8')
            self.roi_deeplearn_show = np.zeros((492, 656, 3), dtype='uint8')
            self.roi = np.zeros((492, 656, 3), dtype='uint8')
            self.img = np.zeros((492, 656, 3), dtype='uint8')
            while (1):
                #新线程
                while_process_singlecam = ThreadWithReturnValue(target=self.While_Process_Singlecam,args=(cam,frame_num,roi_deeplearn))
                while_process_singlecam.start()

                length = len(threading.enumerate())
                print('线程数 ', length)
                
                self.predictresult, self.red_ratio, self.img,self.roi_deeplearn_show, self.roi, frame_num = while_process_singlecam.join()






                # 由后续Roi_Area计算
                if self.predictresult != None:
                    self.label_defectkinds_1.setText("预测种类：" + str(self.predictresult))

                self.label_Picture_1capture.setPixmap(self.roi_deeplearn_show)  # 图像显示在label上

                self.label_redratio_1.setText("红色比例：" + str(self.red_ratio) + "%")

                self.label_Picture_1.setPixmap(self.img)

                self.label_Picture_1process.setPixmap(self.roi)

                self.label_framenum.setText("当前帧：" + str(frame_num))

                QApplication.processEvents()  # 刷新界面显示




                if self.close1:  # 判断是否按下关闭

                    blackimg = np.zeros((492, 656, 3), dtype='uint8')
                    blackimg = QImage(blackimg[:], blackimg.shape[1], blackimg.shape[0], blackimg.shape[1] * 3,
                                      QImage.Format_RGB888)
                    blackimg = QPixmap(blackimg).scaled(self.label_Picture_1.width(), self.label_Picture_1.height())
                    self.label_Picture_1.setPixmap(blackimg)
                    self.label_Picture_1process.setPixmap(blackimg)
                    self.close1 = False
                    break
                else:
                    pass

    def closecamera(self):
        pass

    def showmessage_error(self):
        reply = QMessageBox().information(self, '提示', '未检测到相机，请连接后重试。',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("OK")
        else:
            print("ok")


class DeepLearning(QMainWindow, Ui_Form, Thread):
    def __init__(self):

        super(DeepLearning, self).__init__()
        self.setupUi(self)
        Thread.__init__(self)
        self.result = self.deeplearning_predict()

        self.btn_close_camera.clicked.connect(self.closecam)
        self.close1 = False

    def closecam(self):
        self.close1 = True

    def load_img(self, path):
        img = paddle.dataset.image.load_and_transform(
            path,  # 路径
            400,  # 缩放大小
            100,  # 裁剪
            False
        ).astype("float32")  # 不是用于训练
        img = img / 255.0
        return img

    def deeplearning_predict(self):

        predict_result_return = 0
        predict_img = []  # 建立一个数组
        predict_img.append(self.load_img("roi.bmp"))  # 将待预测模型读入数组
        predict_img = np.array(predict_img)  # 转化为数组
        predict_result = infer_exe.run(infer_program,  # 预测program
                                       feed={feed_names[0]: predict_img},  # 喂入参数
                                       fetch_list=fetch_targets)  # 返回推理结果
        r = np.argmax(predict_result[0])
        for result, kk in predict_dict.items():  # vv是类名，kk是类别数，max是概率
            if kk == r:  # 如果字典中第二位与当前值相同
                max = float(np.max(predict_result)) * 100  # 给出预测比例
                max = ('%.2f' % max)
                print(kk)
                print(predict_result[0])
                """cv2.putText(roi, "Predict result is " +  # 在图像界面上显示
                            str(kk) +
                            " with confidence " +
                            str(max) + "%", (0, 28),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255, 255, 255), 2)
                cv2.putText(roi, "total num: " +
                            str(count_num), (0, 300),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)"""
                predict_result_return = result
                print("Predict result is ", str(result))
                # self.label_defectkinds.setText(result)
                # QMainWindow.label_redratio_1.setText("What Fuck")
                QApplication.processEvents()  # 刷新界面显示
        return predict_result_return
        # cv2.imshow("The picture has been captured to predict", roi)
        # cv2.waitKey()


class ThreadShow(Thread, Ui_Form, QMainWindow):
    def __init__(self):
        super(ThreadShow, self).__init__()

    def run(self):
        super(ThreadShow, self).__init__()
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        if dev_num == 0:
            print("Number of enumerated devices is 0")
            self.showmessage_error()

        # open the first device
        cam1 = device_manager.open_device_by_index(1)
        cam2 = device_manager.open_device_by_index(2)

        # set continuous acquisition
        cam1.TriggerMode.set(gx.GxSwitchEntry.OFF)
        cam2.TriggerMode.set(gx.GxSwitchEntry.OFF)
        # set exposure
        cam1.ExposureTime.set(10000.0)
        cam2.ExposureTime.set(10000.0)
        # set gain
        cam1.Gain.set(100.0)
        cam2.Gain.set(100.0)

        cam1.stream_on()
        cam2.stream_on()

        while (1):
            # get raw image
            raw_image1 = cam1.data_stream[0].get_image()
            raw_image2 = cam2.data_stream[0].get_image()

            if raw_image1 is None:
                print("Getting image failed.")
            if raw_image2 is None:
                print("Getting image failed.")

            # get RGB image from raw image
            rgb_image1 = raw_image1.convert("RGB")
            rgb_image2 = raw_image2.convert("RGB")

            # improve image quality
            # rgb_image1.image_improvement(color_correction_param, contrast_lut, gamma_lut)
            # rgb_image2.image_improvement(color_correction_param, contrast_lut, gamma_lut)

            # create numpy array with data from raw image
            numpy_image1 = rgb_image1.get_numpy_array()
            numpy_image2 = rgb_image2.get_numpy_array()

            # 相机1
            frame1 = White_balance(numpy_image1)

            frame1 = QImage(frame1[:], frame1.shape[1], frame1.shape[0], frame1.shape[1] * 3,
                            QImage.Format_RGB888)  # 将图像转化为Qimage
            img1 = QPixmap(frame1).scaled(self.label_Picture.width(), self.label_Picture.height())
            self.label_Picture.setPixmap(img1)

            # 相机2
            frame2 = White_balance(numpy_image2)
            frame2 = QImage(frame2[:], frame2.shape[1], frame2.shape[0], frame2.shape[1] * 3,
                            QImage.Format_RGB888)  # 将图像转化为Qimage
            img2 = QPixmap(frame2).scaled(self.label_Picture_2.width(), self.label_Picture_2.height())
            self.label_Picture_2.setPixmap(img2)
            QApplication.processEvents()

    def showmessage_error(self):
        reply = QMessageBox().information(self, '提示', '未检测到相机，请连接后重试。',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("OK")
        else:
            print("ok")
    # 按间距中的绿色按钮以运行脚本。


if __name__ == '__main__':
    totalnum = 0
    CapArea = 0
    count_num = 0
    # 深度学习内容，定义执行器
    place = fluid.CPUPlace()
    infer_exe = fluid.Executor(place)
    model_save_dir = "D:\\VGG-fruit-detection\\fruit\\20210510"
    # 加载模型
    infer_program, feed_names, fetch_targets = \
        fluid.io.load_inference_model(model_save_dir,
                                      infer_exe)
    predict_dict = {'apple': 0, 'banana': 1, 'orange': 2, 'mango': 3, "maze": 4, "kiwi": 5}  # 可修改

    app = QApplication(sys.argv)
    gui = Gui615()
    gui.show()

    sys.exit(app.exec_())
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
