import time
import face_recognition
import cv2
import numpy as np
import json
import os
import wave
import pyaudio
import subprocess
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.externals.joblib import dump, load
import sys
from keras.models import model_from_json

def GetFile():
	path = "dataset/"
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
#Convert to Audio Format
def ConvertToWav(file_name,file):
	test = subprocess.Popen(["ffmpeg","-y","-hide_banner","-loglevel","panic","-i",file_name, 'AUDIO/audio/'+file+'.wav', ], stdout=subprocess.PIPE)
	cv2.waitKey(1000)

#Euclidian Distance
def distance(a,b):
	x1,y1 = a
	x2,y2 = b
	return math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
#Functions for FFCOLOR
def Distance(a,b):
	return math.sqrt(pow(a[0]-b[0],2)+ pow(a[1]-b[1],2))
def DDistance(a,b):
	return math.sqrt(pow(a[0]-b[0],2)+ pow(a[1]-a[1],2))
def SDistance(a,b):
	return math.sqrt(pow(b[0]-b[0],2)+ pow(a[1]-b[1],2))
def GetCenter(a):
	global detected_face
	return detected_face[a][int(len(detected_face[a])/2)]
def Color(pt):
	global img
	#img1
	img1 = cv2.copyMakeBorder(
				 img,
				 300,
				 300,
				 300,
				 300,
				 cv2.BORDER_CONSTANT,
				 value=(255,255,255)
			  )
	#cv2.imshow("img",img1)
	#cv2.waitKey(10000)
	img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	return img2[pt[0],pt[1]]
def Point(pt):
	global pixel
	#print(pt)
	try:
		pixel.append(int(Color(pt)))
	except Exception as exceptions:
		#print("In point "+ str(exceptions))
		return True
	return False

def toggle(a):
	if(a==0):
		return 1
	else:
		return 0
# Initializing the PyAudio
chunk = 1024
p = pyaudio.PyAudio()

#The path for the dataset

path = "dataset/"

#req = os.listdir(path)
req, F = GetFile()
video_number = 0
len_req = str(len(req))
real = 0
tr = 0
fake = 0
tf = 0
number = 0
csv = np.array([1,2,3,4,5])
print(csv)
F_np = np.array(F)
np.savetxt("F.csv",F_np,delimiter=",",fmt='% 1.3f')
for file in req:
	#For #printing file number
	video_number+=1
	if(file=="metadata.json"):
		continue
	print(str(video_number)+"/"+len_req+" "+str(F[video_number-1]))
	current_file = path+"/"+file
	ConvertToWav(current_file,file)
	cap = cv2.VideoCapture(current_file)
	audio_file = 'audio/audio/'+file+'.wav'
	while not os.path.exists(audio_file):
		time.sleep(1)
	wf = wave.open(audio_file, 'rb')
	fn = 1
	current_frame = 0
	done = 0
	while(cap.isOpened()):
		##print(done)
		data = wf.readframes(chunk)
		ret,img=cap.read()
		try:
			img.all
		except:
			#print("No face found")
			break
		current_frame+=1
		if(done):
			done = 0
			break
		if(current_frame==fn):
			##print(current_frame)
			try:

				detected_face = face_recognition.face_landmarks(img)[0]

				#AUDIO BEGIN
				chin=detected_face['nose_tip'][int(len(detected_face['nose_tip'])/2)]
				l = []
				left_channel = data[0]
				volume = np.linalg.norm(left_channel)
				for x,y in zip(detected_face['top_lip'],detected_face['bottom_lip']):
					l.append(distance(y,x))
				l.append(volume)
				audio = np.array([l])
				##print(audio)
				json_file = open('AUDIO/model.json', 'r')
				audio_model_json = json_file.read()
				json_file.close()
				audio_model = model_from_json(audio_model_json)
				# load weights into new model
				audio_model.load_weights("AUDIO/model.h5")
				#print("Loaded Audio Model!")
				#classifier.compile(optimizer = 'RMSprop', loss = 'mean_squared_error', metrics = ['accuracy'])
				audio_model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
				sc_audio = load('AUDIO/Audo_sc.bin')
				audio = sc_audio.transform(audio)
				audio_predicted = audio_model.predict(audio)
				audio_weight = math.sqrt(keras.losses.mean_squared_error((audio),(audio_predicted)))
				#print("" + str(audio_weight))
				#AUDIO END

				#REGION BEGIN

				face = face_recognition.face_locations(img)
				if(str(face)=="[]"):
					continue
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
				temp1 = [temp[0][0][0],temp[0][1][0],temp[1][0][0],temp[1][1][0]]
				#print(region)
				region = np.array([])
				for x in range(1,len(temp1),1):
					k = math.sqrt(keras.losses.mean_squared_error([temp1[x-1]],[temp1[x]]))
					#if(str(region)=="[]"):
					region = np.append(region,k)
					#else:
					#	region=np.a[[e((region,k))
				#p = np.vstack((p,F[number-1]))
				#print(region)

				json_file = open('REGION/model.json', 'r')
				region_model_json = json_file.read()
				json_file.close()
				region_model = model_from_json(region_model_json)
				# load weights into new model
				region_model.load_weights("REGION/model.h5")
				#print("Loaded Audio Model!")
				#classifier.compile(optimizer = 'RMSprop', loss = 'mean_squared_error', metrics = ['accuracy'])
				region_model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
				sc_region = load('REGION/REGION_sc.bin')
				region = sc_region.transform([region])
				region_predicted = region_model.predict(region)
				region_weight = math.sqrt(keras.losses.mean_squared_error((region),(region_predicted)))
				#Region Weights
				#print(""+str(region_weight))
				#REGION END

				#FFCOLOR BEGIN
				nose = detected_face['nose_tip'][int(len(detected_face['nose_tip'])/2)]
				eye=detected_face['left_eyebrow'][int(len(detected_face['left_eyebrow'])/2)]
				pixel = []
				push  = 10
				count = 0
				#print(len(detected_face['left_eyebrow']))
				#print(len(detected_face['right_eyebrow']))
				#print(len(detected_face['bottom_lip']))
				for x in detected_face['left_eyebrow']: #Getting 2
					count+=1

					if(count%2==0):
						continue
					if(Point((abs(int(DDistance(nose,x)+push-nose[0])),abs(int(SDistance(nose,x)+push-nose[1]))))):
						count-=1

				push  = 10
				count = 0
				for x in detected_face['right_eyebrow']:# Getting 2
					count+=1

					if(count%2==0):
						continue
					if(Point((abs(int(DDistance(nose,x)+push+nose[0])),abs(int(SDistance(nose,x)+push-nose[1]))))):
						count-=1

				push  = 10
				count = 0
				count2 = 0
				for x in detected_face['bottom_lip']:#getting 8

					count+=1
					if(count%2==0):
						continue
					count2+=1
					if(count2%3==0):
						continue
					if(x[0]<=nose[0]):
						if((Point((abs(int(DDistance(nose,x)+push-nose[0])),abs(int(SDistance(nose,x)+push+nose[1])))))):
							count-=1
					if((Point((abs(int(DDistance(nose,x)+push+nose[0])),abs(int(SDistance(nose,x)+push+nose[1])))))):
						count-=1
				#print(pixel)
				json_file = open('FFCOLOR/model.json', 'r')
				ffcolor_model_json = json_file.read()
				json_file.close()
				ffcolor_model = model_from_json(ffcolor_model_json)
				# load weights into new model
				ffcolor_model.load_weights("FFCOLOR/model.h5")
				sc_ffcolor = load('FFCOLOR/Ffcolor_sc.bin')
				ffcolor = np.array([pixel])
				ffcolor = sc_ffcolor.transform(ffcolor)
				ffcolor_predicted = ffcolor_model.predict(ffcolor)
				ffcolor_weight = math.sqrt(keras.losses.mean_squared_error((ffcolor),(ffcolor_predicted)))
				#FFCOLOR Weights
				#print("FFCOLOR Weights "+str(ffcolor_weight))
				#FFCOLOR END
				#CNN BEGIN
				json_file = open("CNN/model.json", 'r')
				CNN_model_json = json_file.read()
				json_file.close()
				CNN_model = model_from_json(CNN_model_json)
				# load weights into new model
				CNN_model.load_weights("CNN/model.h5")
				CNN_Image = cv2.resize(face_image, (64, 64), interpolation=cv2.INTER_LINEAR)
				#CNN_Image = cv2.cvtColor(CNN_Image, cv2.COLOR_BGR2GRAY)
				#face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
				CNN_Image = np.expand_dims(CNN_Image, axis=0)
				CNN_weight = CNN_model.predict([CNN_Image])
				#CNN Weights
				#print("CNN Weights"+str(toggle(int(float(str(CNN_weight[0][0]))))))
				#print(audio_weight)
				#print(region_weight)
				#print(ffcolor_weight)
				this = [audio_weight,region_weight,ffcolor_weight,toggle(int(float(str(CNN_weight[0][0])))),F[video_number-1]]
				csv = np.vstack((csv,))
				number+=1
				np.savetxt("mydataMAIN/"+str(number)+".csv",csv,delimiter=",",fmt='% 1.3f')
				#print(csv)
				done = 1

				#All Weights
				json_file = open("model.json",'r')
				MAIN_model_json = json_file.read()
				json_file.close()
				MAIN_model = model_from_json(MAIN_model_json)
				MAIN_model.load_weights("model.h5")
				Final_weight = MAIN_model.predict(this)
				print(Final_weight)
				print("Good!")
			except Exception as exceptions:
				exc_type, exc_obj, exc_tb = sys.exc_info()

				print(exceptions,exc_tb.tb_lineno)
				fn+=50
				continue
np.savetxt("main.csv",csv,delimiter=",",fmt='% 1.3f')
