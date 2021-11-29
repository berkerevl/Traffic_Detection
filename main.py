import cv2
import uuid
import time
import os
import datetime

#--------------------------------Random Name Generator-------------------------------#
def RandNameGenerator():
	filename = str(uuid.uuid4())
	return filename[0:8]
#-----------------------------End of RandNameGenerator() ----------------------------#



#-------------------------------Capture the Image------------------------------------#
def CaptureImage(name):
	cam = cv2.VideoCapture(0)
	cam.set(3,800)
	cam.set(4,600)
	
	cv2.namedWindow("Image")
	img_counter = 0
	
	ret,frame = cam.read()
	if not ret:
		print("Failed to grab frame")
	cv2.imshow("test",frame)	
		
	time.sleep(4)
	img_name = name + ".png".format(img_counter)
	cv2.imwrite(img_name,frame)
	print("Succesful")
	img_counter+=1
	time.sleep(2)
	
	cam.release()	
	cv2.destroyAllWindows()
#-----------------------------End of CaptureImage()----------------------------------#


	
#----------------------------Delete the image file-----------------------------------#
def DeleteImage(name):
	cwd = os.getcwd()
	file_path = cwd + "/"+ name + ".png"
	#print(file_path)
	os.remove(file_path)
#-----------------------------End of DeleteImage()-----------------------------------#



#----------------------------Object Detection----------------------------------------#	
def ObjectDetect(name):
	cam = cv2.VideoCapture(0)
	cam.set(3,800)
	cam.set(4,600)
	
	img_nameX = name + ".jpeg"
	img = cv2.imread(img_nameX)
	
	classNames = []
	classFile = 'coco.names'
	with open(classFile,'rt') as f:
		classNames = f.read().rstrip('\n').split('\n')
		
	# Paths
	configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
	weightsPath = 'frozen_inference_graph.pb'

	net = cv2.dnn_DetectionModel(weightsPath,configPath)
	net.setInputSize(480,480)
	net.setInputScale(1.0/127.5)
	net.setInputMean((127.5,127.5,127.5))
	net.setInputSwapRB(True)

	success, img = cam.read()
	classIds, confs, bbox = net.detect(img,confThreshold=0.5)
	num = len(classIds)
		
	if num != 0:
		for classIds, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
			cv2.rectangle(img,box,color=(0,0,255),thickness=1)
			cv2.putText(img,classNames[classIds-1].upper(),(box[0]+10,box[1]+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
		print ("{} {}".format(num,"object detected"))
	else:
		print("0 object detected")			
		
	return num
	
	#cv2.imshow("Output",img)	
#-----------------------------End of ObjectDetect()----------------------------------#



#-----------------------------Density Calculate--------------------------------------#
def DensityCalculate(VehNum):
	return ((VehNum*5)/100)	
#-----------------------------End of DensityCalculate()------------------------------#



#-----------------------------End of ToTextFile()----------------------------------#
def ToTextFile(num,pnpf):

#I will change the numbers and static data and I'll use next() to go next??????????????????????

	while pnpf != 0:
		file_object = open("output.txt","a")
		f.write("001\t" + "Binghamton\t" + "13906\t" + "001\t" + "State st\t" + str(num) + "\t"+ str(datetime.datetime.now()) + "\n")
		pnpf-=1		
#-----------------------------End of ToTextFile()------------------------------#



#------------------------------------------------------------------------------------#
#----------------------------End Of All Functions------------------------------------#
#------------------------------------------------------------------------------------#



#-----------------------------The Code Begins----------------------------------------#

#To create random image name
#To prevent images created later from having the same name in case of an error
rand_name = RandNameGenerator()

#To create output file
f = open("output.txt","w+")

#The number of photo number in one flight (photo number per flight)
PNPF = 1

#to find the location of txt file to delete it
cwd = os.getcwd()
file_path = cwd + "/output.txt"


#the number will be modified according to the time it takes for the drone to reach the location
time.sleep(1)

CaptureImage(rand_name)
data = DensityCalculate(ObjectDetect(rand_name))
ToTextFile(data,PNPF)

DeleteImage(rand_name)
f.close()

#We can use something else here to delete.(Signal from databas notification that notify to data collected)
#delete the output.txt
time.sleep(1) #sleep time to get the data from output to database
os.remove(file_path)


