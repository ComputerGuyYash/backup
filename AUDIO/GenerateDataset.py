import time
import face_recognition
import cv2
import numpy as np
import json
import os
import wave
import pyaudio
import matplotlib.pyplot as plt
import subprocess
import math
# SetUpPyAudio(audio_file):
chunk = 1024
p = pyaudio.PyAudio()

    def ConvertToWav(file_name,file):

        # ffmpeg\ffmpeg-20200101-7b58702-win64-static\bin
        test = subprocess.Popen(["ffmpeg","-y","-hide_banner","-loglevel","panic","-i",file_name, 'audio/'+file+'.wav', ], stdout=subprocess.PIPE)
        # test = subprocess.Popen(["ffmpeg","-i",file_name, 'audio/'+file+'.wav', ], stdout=subprocess.PIPE)
        cv2.waitKey(1000)
    #command =f'ffmpeg\\ffmpeg-20200101-7b58702-win64-static\\bin\\ffmpeg.exe -i '+file_name+' audio/'+file+'.wav'
    #print(command)
    #os.system(command)
p = pyaudio.PyAudio()
path = "dataset/dataset"
req = []
with open(path+"/metadata.json") as json_file:
    data = json.load(json_file)
    #print(data)
    for x in data:
        if(data[x]["label"]=="REAL"):
            req.append(x)
milestone = 0
file = 0
#req=["aamjfukxwp.mp4"]
frames = 0
current_frame = 0
a = np.array([])
def distance(a,b):
    x1,y1 = a
    x2,y2 = b
    return math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
number = 0
for file in req:
    number+=1
    print(file,end = "\t")
    print(str(number)+"/"+str(len(req)))
    current_file = path+"/"+file
    ConvertToWav(current_file,file)
    cap = cv2.VideoCapture(current_file)
    audio_file = 'audio/'+file+'.wav'
    while not os.path.exists(audio_file):
        time.sleep(1)
    try:
        wf = wave.open(audio_file, 'rb')
    except:
        continue

    while(cap.isOpened()):
        data = wf.readframes(chunk)
        ret,img=cap.read()
        if(frames==current_frame):
            frames+=60
            

            scale_percent=100
            try:
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                #cv2.imshow("image",resized)
                left_channel = data[0]
                volume = np.linalg.norm(left_channel)
                #print(volume)
                detected_face = face_recognition.face_landmarks(resized)[0]
                chin=detected_face['nose_tip'][int(len(detected_face['nose_tip'])/2)]
                l = []
                for x,y in zip(detected_face['top_lip'],detected_face['bottom_lip']):
                    l.append(distance(y,x))
                l.append(volume)
                if str(a)=="[]":
                    a = np.append(a,l)
                a = np.vstack((a,l))
                #a = np.append(a,l)
            except Exception as exception:
                np.savetxt("mydataAUDIO/"+str(milestone)+".csv",a,delimiter=",",fmt='% d')
                milestone+=1
                break
                #print(exception)
                #current_frame+=10
                #frames+=10
                #pass
        current_frame+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
np.savetxt("mydataAUDIO/end.csv",a,delimiter=",",fmt='% d')

