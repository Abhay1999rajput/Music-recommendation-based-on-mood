import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
import pandas as pd
import time
train_X = pd.read_csv("DataSet/muse_v3.csv", delimiter=',')
predictedMusic=train_X['lastfm_url']
model = model_from_json(open("model_architecture.json", "r").read())
model.load_weights('model_weights.h5')

classifier =load_model('model_78.h5')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
start=time.time()
last=time.time()
totalEmotions=np.array([0, 0, 0, 0, 0, 0, 0])
while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    faces_detected = faceCascade.detectMultiScale(gray_img, 1.32, 5)
    if(time.time()-start>10):
        e=['Fear', 'Disgust', 'Angry', 'Happy', 'Sad','Neutral','Surprise']
        moodvsmusic =np.array([[-0.64,0.6,-0.43],[-0.6,0.35,0.11],[-0.43,0.67,0.34],
                     [0.76,0.48,0.35], [-0.63,0.27,-0.33],[0.42,0.47,0.12],[0.4,0.67,-0.13]])
        musicnn=[]
        for i in range(len(totalEmotions)):
            t=[]
            for j in range(3):
                t.append(i*float(moodvsmusic[i][j]))
            musicnn.append(t)
        musicFeture=np.mean(np.array(musicnn), axis=0)
        print(musicFeture[0])
        c=np.argmax(totalEmotions)
        pre=[float("{:.2f}".format(musicFeture[0])),float("{:.2f}".format(musicFeture[1])),float("{:.2f}".format(musicFeture[2]))]
        p=[]
        p.append(pre)
        x=model.predict(p)
        print("Please listen to this song",predictedMusic[int(x)])
        # print("average best mood",e[c])
        # music_link=[
        #     "https://music.youtube.com/playlist?list=OLAK5uy_lU7E-YMryMwbpWQvYniEsOL9H1UpTZeSU",
        #     "https://music.youtube.com/playlist?list=OLAK5uy_l6BRlbwM4PIs_BkQcv8zfbAPcgCsjATAE",
        #     "https://music.youtube.com/playlist?list=PL_MH8gOS_ETiNT1NF8B46JYHZe6fXWfVW",
        #     "https://music.youtube.com/playlist?list=RDCLAK5uy_mfdqvCAl8wodlx2P2_Ai2gNkiRDAufkkI",
        #     "https://music.youtube.com/playlist?list=RDCLAK5uy_kCFyity-5xsBCaEQbpfJz8Gxp0zz6eRQ8",
        #     "https://music.youtube.com/watch?v=OcmcptbsvzQ",
        #     "https://music.youtube.com/watch?v=youYVBAuaEM"
        # ]
        # print("Please check these link",music_link[c])
        start=time.time()
        totalEmotions=np.array([0, 0, 0, 0, 0, 0, 0])

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray = gray_img[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            predictions = classifier.predict(roi)[0]
            max_index = np.argmax(predictions)
            e=['Fear', 'Disgust', 'Angry', 'Happy', 'Sad','Neutral', 'Surprise' ]
            emotions = ('Fear', 'Disgust', 'Angry', 'Happy', 'Sad', 'Neutral','Surprise' )
            predicted_emotion = emotions[max_index]
            if(time.time()-last>1):
                predict=np.argmax(predictions)
                totalEmotions[max_index]+=1
                last=time.time()
                # print(totalEmotions)
                # print(e[predict])

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,0), 2)

        # predictions = model.predict(img_pixels)
        
   
        
    resized_img = cv2.resize(test_img, (400,400))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(100) == ord('q'): #wait until 'q' key is pressed
        if cv2.waitKey(): 
            break

cap.release()
cv2.destroyAllWindows