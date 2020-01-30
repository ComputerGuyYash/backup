from sklearn.metrics import mean_squared_error
import face_recognition
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import math
def GetFile():
    path = "dataset/dataset"
    req = []
    F = []
    with open(path+"/metadata.json") as json_file:
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
path = "dataset/dataset"
number = 0
plt.show()
#req = ["aajrvbynqc.mp4"]
s = np.array([])
for file in req:
    number+=1
    file_name = path+"/"+file
    print(file+"\t"+str(number)+"/"+str(len(req)))
    #print(file_name)
    cap = cv2.VideoCapture(file_name)
    cv2.waitKey(1000)
    frame = 1
    ret,img = cap.read()
    #cv2.imshow("Image",img)
    cv2.waitKey(1000)
    frame  = 0
    color = np.array([])
    done = 0
    tframe = 5
    while(cap.isOpened()):
        frame+=1
        if(frame%200!=0):
            continue
        ret,img = cap.read()
        try:
           img.all()
        except:
           break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print("reading")
        try:
            img.all()
        except:
            break
        #image = face_recognition.load_image_file("your_file.jpg")
        face = face_recognition.face_locations(img)
        if(str(face)=="[]"):
            continue
        #print(face)

        done+=1
        if(done==tframe):
            break
        
        #print(temp)
        face_location=face[0]
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = img[top:bottom, left:right]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        #plt.imshow(face_image)
        #plt.show()
        width, height,_ = face_image.shape
        w, h = (2, 2)

        # Resize input to "pixelated" size
        temp = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_LINEAR)
        if(str(color)=="[]"):
            color = np.append(color,[temp[0][0][0],temp[0][1][0],temp[1][0][0],temp[1][1][0]])
        else:
            color = np.vstack((color,[temp[0][0][0],temp[0][1][0],temp[1][0][0],temp[1][1][0]]))
        # Initialize output image
        #output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        #plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        #plt.draw()
        #plt.show()
        #plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        #plt.show()
        #cv2.waitKey(1000)
    if(done!=tframe):
        continue
    #print(color)
    #color=(color/255)
    #for x in range(len(color)):
    #    color[x]=color[x]/255
    p = np.array([])
    for x in range(1,len(color),1):
        k = math.sqrt(mean_squared_error(color[x-1],color[x]))
        if(str(p)=="[]"):
            p = np.append(p,k)
        else:
            p=np.vstack((p,k))
    p = np.vstack((p,F[number-1]))
    cap.release()
    #print(p)
    if(str(s)=="[]"):
        s = np.append(s,p)
    else:
        p = p.reshape(1 ,4)
        s = np.vstack((s,p))
    #print(s)
    np.savetxt("mydataREGION/"+str(number)+".csv",s,delimiter=",",fmt='% 1.3f')
np.savetxt("mydataREGION/end.csv",s,delimiter=",",fmt='% 1.3f')

