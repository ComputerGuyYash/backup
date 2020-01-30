from sklearn.metrics import mean_squared_error
import face_recognition
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import math
def GetFile():
    path = "dataset\\dataset"
    req = []
    F = []
    with open(path+"\\metadata.json") as json_file:
        data = json.load(json_file)
        #print(data)
        for x in data:
            req.append(x)
            if(data[x]["label"]=="FAKE"):
                F.append(1)
            else:
                F.append(0)
    return (req,F)

req,F = GetFile()
path = "dataset\\dataset"
number = 0
plt.show()
#req = ["aajrvbynqc.mp4"]
s = np.array([])
encode = 0
fake = 0
real = 0
for file in req:
    number+=1
    file_name = path+"/"+file
    print(file+"\t"+str(number)+"/"+str(len(req)))
    #print(file_name)
    cap = cv2.VideoCapture(file_name)

    frame = 1
    ret,img = cap.read()
    #cv2.imshow("Image",img)
    #cv2.waitKey(1000)
    frame  = 0
    color = np.array([])
    done = 0
    tframe = 5
    while(cap.isOpened()):
        frame+=1
        #print(frame)
        if(frame%500!=0):
            continue

        ret,img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print("reading")
        try:
            img.all()
        except:
            break

        face = face_recognition.face_locations(img)
        if(str(face)=="[]"):
            break
        face_location=face[0]

        top, right, bottom, left = face_location
        face_image = img[top:bottom, left:right]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        width, height,_ = face_image.shape
        w, h = (64, 64)

        temp = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_LINEAR)
        if(F[number-1]):

            cv2.imwrite("mydataCNN/FAKE/"+str(fake)+".jpg",temp)

            fake+=1
        else:
            cv2.imwrite("mydataCNN/REAL/"+str(real)+".jpg",temp)
            real+=1
    cap.release()
