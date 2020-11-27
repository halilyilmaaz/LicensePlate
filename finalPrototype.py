#from darkflow.net.build import TFNet
#import TFNet
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import imutils
from keras.models import Sequential, load_model


#options = {"pbLoad": "yolo-plate.pb", "metaLoad": "yolo-plate.meta", "gpu": 0.9}
#yoloPlate = TFNet(options)
#
#options = {"pbLoad": "yolo-character.pb", "metaLoad": "yolo-character.meta", "gpu":0.9}
#yoloCharacter = TFNet(options)


classesFile = 'obj.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    


modelConfig = 'yolov3-tiny_obj.cfg'
modelWeights = 'yolov3-tiny_obj_last.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

characterRecognition = load_model('model_char_recognition.h5')

##load model
#model = model_from_json(open("fer.json", "r").read())
##load weights
#model.load_weights('fer.h5')

whT = 320
confTheresholds = 0.5
nmsThereshold = 0.3

def firstCrop(img, predictions):
#    predictions.sort(key=lambda x: x.get('confidence'))
#    xtop = predictions[-1].get('topleft').get('x')
#    ytop = predictions[-1].get('topleft').get('y')
#    xbottom = predictions[-1].get('bottomright').get('x')
#    ybottom = predictions[-1].get('bottomright').get('y')
#    firstCrop = img[ytop:ybottom, xtop:xbottom]
#    cv2.rectangle(img,(xtop,ytop),(xbottom,ybottom),(0,255,0),3)
    hT,wT,cT = img.shape
    bbox = []  
    classIds = []
    confs = []
    for outputs in predictions:
        for det in outputs:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confTheresholds:
                w,h = int(det[2]*wT) ,int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))            
#    print(len(bbox))           
    indices = cv2.dnn.NMSBoxes(bbox,confs,confTheresholds,nmsThereshold) 
#    print(indices)
#    fileName='deneme.jpg'
#    ss=True
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper() } %{int(confs[i]*100)}',
                               (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
  
    
    
def secondCrop(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        secondCrop = img[y:y+h,x:x+w]
    else: 
        secondCrop = img
    return secondCrop

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def opencvReadPlate(img):
    h, w = frame.shape[:2]   
    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), 270, .5)
    img = cv2.warpAffine(img, rotation_matrix, (w, h))
    charList=[]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
    edges = auto_canny(thresh_inv)
    ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = img.shape[0]*img.shape[1]

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w*h
        non_max_sup = roi_area/img_area

        if((non_max_sup >= 0.015) and (non_max_sup < 0.09)):
            if ((h>1.2*w) and (3*w>=h)):
                char = img[y:y+h,x:x+w]
                charList.append(cnnCharRecognition(char))
                cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.imshow('OpenCV character segmentation',img)
    licensePlate="".join(charList)
    return licensePlate

def cnnCharRecognition(img):
    dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
    11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
    21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
    30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}
    im_path="../LicensePlateRecognition/vid1.MOV"
    img = cv2.imdecode(np.fromfile(im_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    blackAndWhiteChar=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackAndWhiteChar = cv2.resize(blackAndWhiteChar,(75,100))
    image = blackAndWhiteChar.reshape((1, 100,75, 1))
    image = image / 255.0
    new_predictions = characterRecognition.predict(image)
    char = np.argmax(new_predictions)
    return dictionary[char]

def yoloCharDetection(predictions,img):

    charList = []
    positions = []
    for i in predictions:
        if i.get("confidence")>0.10:
            xtop = i.get('topleft').get('x')
            positions.append(xtop)
            ytop = i.get('topleft').get('y')
            xbottom = i.get('bottomright').get('x')
            ybottom = i.get('bottomright').get('y')
            char = img[ytop:ybottom, xtop:xbottom]
            cv2.rectangle(img,(xtop,ytop),( xbottom, ybottom ),(255,0,0),2)
            charList.append(cnnCharRecognition(char))

    cv2.imshow('Yolo character segmentation',img)
    sortedList = [x for _,x in sorted(zip(positions,charList))]
    licensePlate="".join(sortedList)
    return licensePlate

cap = cv2.VideoCapture('vid1.MOV')
counter=0

while(cap.isOpened()):
    ret, frame = cap.read()
    
#    h, w, l = frame.shape
    frame = imutils.rotate(frame, 270)
#    frame = cv2.transpose(frame)
#    height, width = frame.shape[:2]   
#    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 270, .5)
#    frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
    if ret:
        assert not isinstance(frame,type(None))

    if counter%6 == 0:
        licensePlate = []
        try:
            
            blob = cv2.dnn.blobFromImage(frame,1/255,(whT,whT),[0,0,0],1,crop=False)
            net.setInput(blob)
            
            layerNames = net.getLayerNames()
        #    print(layerNames)
            
            outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
        #    print(net.getUnconnectedOutLayers())
            
            outputs = net.forward(outputNames)
#            print(outputs[0].shape)
#            print(outputs[1].shape)
        #    print(outputs[2].shape)
            
            
            
            predictions = firstCrop(frame,outputs)
            firstCropImg = firstCrop(frame, predictions)
            secondCropImg = secondCrop()
            cv2.imshow('Second crop plate',secondCropImg)
            secondCropImgCopy = secondCropImg.copy()
            licensePlate.append(opencvReadPlate(secondCropImg))
            print("OpenCV+CNN : " + licensePlate[0])
        except:
            pass
#        try:
#            predictions = yoloCharacter.return_predict(secondCropImg)
#            licensePlate.append(yoloCharDetection(predictions,secondCropImgCopy))
#            print("Yolo+CNN : " + licensePlate[1])
#        except:
#            pass	

    counter+=1
    cv2.imshow('Video',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

