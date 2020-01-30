import cv2
import json
from includer import *
import face_recognition
import math
import numpy as np
req = GetFile("REAL")
path = "dataset/dataset"
def Distance(a,b):
	return math.sqrt(pow(a[0]-b[0],2)+ pow(a[1]-b[1],2))
def DDistance(a,b):
	return math.sqrt(pow(a[0]-b[0],2)+ pow(a[1]-a[1],2))
def SDistance(a,b):
	return math.sqrt(pow(b[0]-b[0],2)+ pow(a[1]-b[1],2))
def GetCenter(a):
	global detected_face
	return detected_face[a][int(len(detected_face[a])/2)]
def Point(pt):
	global pixel
	#print(pt)
	global img
	#img = cv2.line(img,pt,pt,(100,255,255),2)
	#pixel.append(color(pt))
	try:
		img = cv2.line(img,pt,pt,(100,255,255),2)
		pixel.append(color(pt))
	except:
		return True
	return False
def color(pt):
	global img
	return img[pt[0],pt[1]]
arr = np.array([])
number = 0
for file in req:
	number+=1
	file_name = path+"/"+file
	cap = cv2.VideoCapture(file_name)
	frame = 0
	print(str(file)+"\t"+str(number)+ "/"+str(len(req)))
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
		"""try:
			img.all()
		except:
			break
		"""
		try:
			detected_face = face_recognition.face_landmarks(img)[0]
		except:
			continue
		nose = detected_face['nose_tip'][int(len(detected_face['nose_tip'])/2)]
		eye = detected_face['left_eyebrow'][int(len(detected_face['left_eyebrow'])/2)]
		#cv2.line(img,nose,(eye[0],nose[1]),(0,0,255),1)
		#Pusha = (int(DDistance(nose,GetCenter('left_eyebrow'))+10),int(SDistance(nose,GetCenter('left_eyebrow'))+10))
		#x2 = x1 + length * cos(θ)
		#y2 = y1 + length * sin(θ) 
		#Point(Pusha)
		#print(math.degrees(math.acos(DDistance(nose,GetCenter('left_eyebrow'))/(Distance(nose,GetCenter('left_eyebrow'))))))
		#print(math.degrees(math.asin(DDistance(nose,GetCenter('left_eyebrow'))/(Distance(nose,GetCenter('left_eyebrow'))))))
		#img = cv2.line(img,nose,GetCenter('left_eyebrow'),(0,255,0),1)
		#ACOS = (math.acos(DDistance(nose,GetCenter('left_eyebrow'))/(Distance(nose,GetCenter('left_eyebrow')))))
		#ASIN = (math.asin(DDistance(nose,GetCenter('left_eyebrow'))/(Distance(nose,GetCenter('left_eyebrow')))))
		pixel = []
		push  = 10
		count=0
		for x in detected_face['left_eyebrow']:
			count+=1
			if(count%2==0):
				continue
			if(Point((abs(int(DDistance(nose,x)+push-nose[0])),abs(int(SDistance(nose,x)+push-nose[1]))))):
				break

		push  = 10
		count = 0
		for x in detected_face['right_eyebrow']:
			count+=1
			if(count%2==0):
				continue
			Point((abs(int(DDistance(nose,x)+push+nose[0])),abs(int(SDistance(nose,x)+push-nose[1]))))
			
		push  = 10
		count = 0
		count2 = 0
		for x in detected_face['bottom_lip']:
			count+=1
			if(count%2==0):
				continue
			count2+=1
			if(count2%3==0):
				continue
			if(x[0]<=nose[0]):
				if((Point((abs(int(DDistance(nose,x)+push-nose[0])),abs(int(SDistance(nose,x)+push+nose[1])))))):
					break
			if((Point((abs(int(DDistance(nose,x)+push+nose[0])),abs(int(SDistance(nose,x)+push+nose[1])))))):
				break
		#push  = 10
		#for x in detected_face['chin']: 
		#	Point(x)
			#Point((abs(int(DDistance(nose,x)+push-nose[0])),abs(int(SDistance(nose,x)+push-nose[1]))))
		#if(str(arr)=='[]'):
		#	arr = np.array(pixel)
		if(len(pixel)==12):
			if(str(arr)=='[]'):
				arr = np.array(pixel)
			else:
				arr = np.vstack((arr,pixel))
		pixel = []
		#cv2.imshow("Video",img)
		#cv2.waitKey(1)
	np.savetxt("mydataCOLOR/"+str(number)+".csv",arr,delimiter=",",fmt='% d')
np.savetxt("mydataCOLOR/end.csv",arr,delimiter=",",fmt='% d')

