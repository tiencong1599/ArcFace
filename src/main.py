import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from imutils import paths
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from softmax import SoftMax
from main_window import *
import sys
import matplotlib.pyplot as plt
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')
from keras.models import load_model
from mtcnn import MTCNN
import face_preprocess
import numpy as np
import face_model
import pickle
import time
import dlib
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_UIT()
        self.ui.setupUi(self)
        self.setWindowTitle("UIT Face Recognition")
        self._mutex = QMutex()
        ############ detail capture
        self.mymodel = 'outputs/my_model.h5'
        self.embeddings_model = 'outputs/embeddings.pickle'
        self.data = pickle.load(open('outputs/embeddings.pickle', 'rb'))
        self.le = pickle.load(open('outputs/le.pickle', 'rb'))
        self.embeddings = np.array(self.data['embeddings'])
        self.labels = self.le.fit_transform(self.data['names'])
        self.outputs = '../datasets/video_output/'
        self.IMAGE_SIZE = '112,112'
        self.MODEL = '../insightface/models/model-y1-test2/model,0'
        self.GPU = 0
        self.DET = 0
        self.FLIP = 0
        self.THRESHOLD = 1.24
        self.args = {'model': self.MODEL,
                'image_size': self.IMAGE_SIZE,
                'ga_model': '',
                'threshold': self.THRESHOLD,
                'det': self.DET}

        self.embedding_model = face_model.FaceModel(self.args)
        self.model = load_model(self.mymodel)
        self.detector = MTCNN()
        self.cosine_threshold = 0.8
        self.proba_threshold = 0.85
        self.comparing_num = 5
        self.trackers = []
        self.texts = []
        self.frames = 0
        self.frames_capture = 0
        self.max_bbox = np.zeros(4)
        self.count = 0
        #############
        # set icon
        self.icon1 = cv2.imread('UIT-logo.jpg')
        self.ui.capture_icon_label.setPixmap(
            QPixmap.fromImage(self.display_image(self.icon1, self.ui.capture_icon_label)))
        self.ui.record_label.setPixmap(QPixmap.fromImage(self.display_image(self.icon1,self.ui.record_label)))
        self.ui.exit_btn.clicked.connect(self.close)

        # create timer for record action
        self.timer_record = QTimer()
        self.timer_record.timeout.connect(self.detectFaces)
        self.ui.record_btn.clicked.connect(self.controlTimer)
        #create timer for capture action
        self.timer_capture = QTimer()
        self.timer_capture.timeout.connect(self.capture)
        self.ui.capture_btn.clicked.connect(self.controlTimerCapture)
        #re-embedding vector
        self.ui.re_embedding_btn.clicked.connect(self.re_embedding)
        #re-train softmax
        self.ui.re_train_btn.clicked.connect(self.re_train)


    def display_image(self, image, element):
        img = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), (element.width(), element.height()))
        qImg = QImage(img.data, img.shape[1], img.shape[0],
                      img.shape[1] * img.shape[2], QImage.Format_RGB888)
        return qImg

    def detectFaces(self):
        def findCosineDistance(vector1, vector2):
            """
            Calculate cosine distance between two vector
            """
            vec1 = vector1.flatten()
            vec2 = vector2.flatten()

            a = np.dot(vec1.T, vec2)
            b = np.dot(vec1.T, vec1)
            c = np.dot(vec2.T, vec2)
            return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

        def CosineSimilarity(test_vec, source_vecs):
            """
            Verify the similarity of one vector to group vectors of one class
            """
            cos_dist = 0
            for source_vec in source_vecs:
                cos_dist += findCosineDistance(test_vec, source_vec)
            return cos_dist / len(source_vecs)

        ret, frame = self.cap.read()
        # resize frame image
        frame = cv2.resize(frame, (self.ui.record_label.width(), self.ui.record_label.height()),
                           interpolation=cv2.INTER_AREA)
        self.frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.frames % 2 == 0:
            trackers = []
            texts = []

            detect_tick = time.time()
            bboxes = self.detector.detect_faces(frame)
            detect_tock = time.time()

            if len(bboxes) != 0:
                reco_tick = time.time()
                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    landmarks = bboxe['keypoints']
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    nimg = np.transpose(nimg, (2, 0, 1))
                    embedding = self.embedding_model.get_feature(nimg).reshape(1, -1)

                    text = "Unknown"

                    # Predict class
                    preds = self.model.predict(embedding)
                    preds = preds.flatten()
                    # Get the highest accuracy embedded vector
                    j = np.argmax(preds)
                    proba = preds[j]
                    # Compare this vector to source class vectors to verify it is actual belong to this class
                    match_class_idx = (self.labels == j)
                    match_class_idx = np.where(match_class_idx)[0]
                    selected_idx = np.random.choice(match_class_idx, self.comparing_num)
                    compare_embeddings = self.embeddings[selected_idx]
                    # Calculate cosine similarity
                    cos_similarity = CosineSimilarity(embedding, compare_embeddings)
                    if cos_similarity < self.cosine_threshold and proba > self.proba_threshold:
                        name = self.le.classes_[j]
                        text = "{}".format(name)
                        print("Recognized: {} <{:.2f}>".format(name, proba * 100))
                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                    cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        self.ui.record_label.setPixmap(QPixmap.fromImage(self.display_image(frame,self.ui.record_label)))

    def controlTimer(self):
        # if timer is stopped
        if not self.timer_record.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer_record.start(2)
            # update control_bt text
            self.ui.record_btn.setText("Stop")

        # if timer is started
        else:
            # stop timer
            self.timer_record.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.record_label.setPixmap(QPixmap.fromImage(self.display_image(self.icon1, self.ui.record_label)))
            self.ui.record_btn.setText("Record")

    def capture(self):
        ret, frame = self.capture.read()
        # Get all faces on current frame
        self.frames_capture +=1
        bboxes = self.detector.detect_faces(frame)
        if len(bboxes) != 0:
            # Get only the biggest face
            max_area = 0
            for bboxe in bboxes:
                bbox = bboxe["box"]
                bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                keypoints = bboxe["keypoints"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > max_area:
                    self.max_bbox = bbox
                    landmarks = keypoints
                    max_area = area
            self.max_bbox = self.max_bbox[0:4]
            landmarks = np.array(
                [landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0],
                 landmarks["mouth_right"][0],
                 landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1],
                 landmarks["mouth_right"][1]]).reshape((2, 5)).T
            nimg = face_preprocess.preprocess(frame, self.max_bbox, landmarks, image_size='112,112')
            cv2.rectangle(frame,(self.max_bbox[0],self.max_bbox[1]),(self.max_bbox[2],self.max_bbox[3]),(255,0,0),1)
            cv2.putText(frame, 'Numbers of Image: {}'.format(self.count), (30, 15), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, 'Time: {}'.format(self.frames_capture).format(), (30, 45), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            if self.count <=5:
                if self.frames_capture % 10==0:
                    #cv2.imwrite('../datasets/unlabeled_faces/{}.jpg'.format(self.frames_capture),nimg)
                    self.trackers.append(nimg)
                    cv2.rectangle(frame, (self.max_bbox[0], self.max_bbox[1]), (self.max_bbox[2], self.max_bbox[3]),
                                  (0, 0, 255), 3)
                    self.count+=1
            else:
                if not os.path.exists('../datasets/train/' + self.name_edit + '-' + self.id_edit):
                    os.mkdir('../datasets/train/' + self.name_edit + '-' + self.id_edit)
                    for i,j in enumerate(self.trackers):
                        cv2.imwrite('../datasets/train/'+self.name_edit+'-'+self.id_edit+'/{}.jpg'.format(i),j)
                else:
                    for i,j in enumerate(self.trackers):
                        cv2.imwrite('../datasets/train/'+self.name_edit+'-'+self.id_edit+'/{}.jpg'.format(i),j)
                QMessageBox.information(self, "Error" , "Đủ số lượng ảnh")
                self.trackers.clear()
                self.capture.release()
                self.timer_capture.stop()
                self.ui.name_txt.clear()
                self.ui.id_txt.clear()
                self.count=0
                self.frames_capture=0
                self.ui.capture_btn.setText("Capture")
            self.ui.capture_icon_label.setPixmap(
                QPixmap.fromImage(self.display_image(frame, self.ui.capture_icon_label)))

    def controlTimerCapture(self):
        # if timer is stopped
        if not self.timer_capture.isActive():
            # create video capture
            self.name_edit = self.ui.name_txt.text()
            self.id_edit = self.ui.id_txt.text()
            if self.id_edit == '' or self.name_edit == '':
                QMessageBox.information(self, "Error", "Không được bỏ trống tên và mã số nhân viên")
            else:
                self.capture = cv2.VideoCapture(0)
                # start timer
                self.timer_capture.start(30)
                # update control_bt text
                self.ui.capture_btn.setText("Stop")

        # if timer is started
        else:
            # stop timer
            self.timer_capture.stop()
            # release video capture
            self.capture.release()
            self.ui.name_txt.clear()
            self.ui.id_txt.clear()
            self.count=0
            self.frames_capture=0
            self.trackers.clear()
            # update control_bt text
            self.ui.capture_icon_label.setPixmap(QPixmap.fromImage(self.display_image(self.icon1, self.ui.record_label)))
            self.ui.capture_btn.setText("Capture")
    def re_embedding(self):
        knownEmbeddings = []
        knownNames = []
        args = {'model': '../insightface/models/model-y1-test2/model,0',
                'image_size': '112,112',
                'ga_model': '',
                'threshold': 1.24,
                'det': 0,
                'dataset': '../datasets/train',
                'embeddings': 'outputs/embeddings.pickle'}

        # Grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(args['dataset']))

        # Initialize the faces embedder
        embedding_model = face_model.FaceModel(args)
        # Initialize the total number of faces processed
        total = 0

        # Loop over the imagePaths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load the image
            image = cv2.imread(imagePath)
            # convert face to RGB color
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2, 0, 1))
            # Get the face embedding vector
            face_embedding = embedding_model.get_feature(nimg)

            # add the name of the person + corresponding face
            # embedding to their respective list
            knownNames.append(name)
            knownEmbeddings.append(face_embedding)
            total += 1

        print(total, " faces embedded")

        # save to output
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(args['embeddings'], "wb")
        f.write(pickle.dumps(data))
        f.close()
        QMessageBox.information(self,"Error","Embedding hoàn tất")
    def re_train(self):
        args = {'embeddings': 'outputs/embeddings.pickle',
                'model': 'outputs/my_model.h5',
                'le': 'outputs/le.pickle'}

        # Load the face embeddings
        data = pickle.loads(open(args["embeddings"], "rb").read())

        # Encode the labels
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)
        one_hot_encoder = OneHotEncoder()
        labels = one_hot_encoder.fit_transform(labels).toarray()

        embeddings = np.array(data["embeddings"])

        # Initialize Softmax training model arguments
        BATCH_SIZE = 32
        EPOCHS = 20
        input_shape = embeddings.shape[1]

        # Build sofmax classifier
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
        model = softmax.build()

        # Create KFold
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
        # Train
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[
                valid_idx]
            his = model.fit(X_train, y_train,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            validation_data=(X_val, y_val))
            print(his.history['accuracy'])

            history['acc'] += his.history['accuracy']
            history['val_acc'] += his.history['val_accuracy']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']

        # write the face recognition model to output
        model.save(args['model'])
        f = open(args["le"], "wb")
        f.write(pickle.dumps(le))
        f.close()
        QMessageBox.information(self,"Error","Train hoàn tất")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())