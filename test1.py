#python drowniness_yawn.py --webcam webcam_index

from scipy.spatial import distance as dist
import tflite_runtime.interpreter as tflite
from imutils.video import VideoStream
from PIL import Image
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import collections
import imutils
import common
import time
import dlib
import cv2
import os
import re
from pose_engine import PoseEngine
Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
c=0


def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))
    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        c = 0
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
       cv2.putText(cv2_im,"PERSON ON WHEEL :", (300,400),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
       cv2.putText(cv2_im,"OBJECT DETECTED :", (300,430),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
       if 'cell phone' in label:
           cv2.putText(cv2_im,"Cell Phone..", (550,430),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,225),2)

    return cv2_im


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
default_model_dir = 'model'
default_labels = 'labels2.txt'
default_model = 'ssd_mobilenet_v1_customtrained_quant_edgetpu.tflite'
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
#parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
args = parser.parse_args()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
engine = PoseEngine('models/mobilenet/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite')


print("-> Starting Video Stream")
#vs = VideoStream("VIDEO_20200918_181318086.mp4").start()
#vs= VideoStream(0).start()
vs=cv2.VideoCapture(0)
print('Loading {} with {} labels.'.format(args.model, args.labels))
interpreter = common.make_interpreter(args.model)
interpreter.allocate_tensors()  
labels = load_labels(args.labels)     
#//For Raspberry Pi
#time.sleep(1.0)

while True:

    ret,frame = vs.read()
    #if not frame:
    #    break
    cv2.putText(frame,"EYES:", (10,400),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    cv2.putText(frame,"YAWN:", (10,430),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    cv2_im = frame
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    common.set_input(interpreter, pil_im)
    interpreter.invoke()    
    objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
    cv2_im = append_objs_to_img(cv2_im, objs, labels)

    frame = imutils.resize(frame, width=840)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    poses, inf = engine.DetectPosesInImage(frame)
    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    
    for pose in poses:
        print("Pose Start")
        for label, keypoint in pose.keypoints.items():
            print(' %-20s x=%-4d y=%-4d' %(label, keypoint.yx[1], keypoint.yx[0]))

   #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (255, 0, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            cv2.putText(frame, "Closed ", (100, 400),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            cv2.putText(frame, "Open ", (100, 430),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            alarm_status = False

        if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (100, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    t = Thread(target=alarm, args=('take some fresh air',))
                    t.deamon = True
                    t.start()
        else:
            alarm_status2 = False


    cv2.imshow("DRIVER MONITORING SYSTEM", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
