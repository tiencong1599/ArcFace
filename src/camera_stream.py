import sys
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


mymodel = 'outputs/my_model.h5'
data = pickle.load(open('outputs/embeddings.pickle','rb'))
le = pickle.load(open('outputs/le.pickle','rb'))
embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])
outputs = '../datasets/video_output/'
IMAGE_SIZE = '112,112'
MODEL = '../insightface/models/model-y1-test2/model,0'
GPU = 0
DET = 0
FLIP = 0
THRESHOLD = 1.24
def camera_open():
    args = {'model':MODEL,
            'image_size':IMAGE_SIZE,
            'ga_model':'',
            'threshold':THRESHOLD,
            'det':DET}

    embedding_model = face_model.FaceModel(args)
    model = load_model(mymodel)
    detector = MTCNN()
    def findCosineDistance(vector1, vector2):
        """
        Calculate cosine distance between two vector
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

    def CosineSimilarity(test_vec, source_vecs):
        """
        Verify the similarity of one vector to group vectors of one class
        """
        cos_dist = 0
        for source_vec in source_vecs:
            cos_dist += findCosineDistance(test_vec, source_vec)
        return cos_dist/len(source_vecs)

    # Initialize some useful arguments
    cosine_threshold = 0.8
    proba_threshold = 0.85
    comparing_num = 5
    trackers = []
    texts = []
    frames = 0

    # Start streaming and recording
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_width = 600
    save_height = int(600/frame_width*frame_height)
    video_out = cv2.VideoWriter(outputs, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (save_width,save_height))

    while True:
        ret, frame = cap.read()
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (save_width, save_height))
        if frames%3 == 0:
            trackers = []
            texts = []

            detect_tick = time.time()
            bboxes = detector.detect_faces(frame)
            detect_tock = time.time()

            if len(bboxes) != 0:
                reco_tick = time.time()
                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                    landmarks = bboxe['keypoints']
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                         landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2,5)).T
                    nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    nimg = np.transpose(nimg, (2,0,1))
                    embedding = embedding_model.get_feature(nimg).reshape(1,-1)

                    text = "Unknown"

                    # Predict class
                    preds = model.predict(embedding)
                    preds = preds.flatten()
                    # Get the highest accuracy embedded vector
                    j = np.argmax(preds)
                    proba = preds[j]
                    # Compare this vector to source class vectors to verify it is actual belong to this class
                    match_class_idx = (labels == j)
                    match_class_idx = np.where(match_class_idx)[0]
                    selected_idx = np.random.choice(match_class_idx, comparing_num)
                    compare_embeddings = embeddings[selected_idx]
                    # Calculate cosine similarity
                    cos_similarity = CosineSimilarity(embedding, compare_embeddings)
                    if cos_similarity < cosine_threshold and proba > proba_threshold:
                        name = le.classes_[j]
                        text = "{}".format(name)
                        print("Recognized: {} <{:.2f}>".format(name, proba*100))
                    # Start tracking
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
                    texts.append(text)

                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                    cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
        else:
            for tracker, text in zip(trackers,texts):
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
                cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255),2)

        cv2.imshow("Frame", frame)
        video_out.write(frame)
        # print("Faces detection time: {}s".format(detect_tock-detect_tick))
        # print("Faces recognition time: {}s".format(reco_tock-reco_tick))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    video_out.release()
    cap.release()
    cv2.destroyAllWindows()

camera_open()