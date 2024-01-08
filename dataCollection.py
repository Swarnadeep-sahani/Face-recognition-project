#read a video from web cam
# face detection in video
#click 20 pictures of  each person 

import cv2
import numpy as np

#create camera object
cam=cv2.VideoCapture(0);
#ask the name 
filename=input("Enter the name of the person:")
dataset_path="./data/"
offset=20
skip=0
cnt=0
#model
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#Create a list of save face data
faceData=[]
cnt=0
#read image from camera
while True:
    success,img=cam.read()
    if not success:
        print("reading camera failure")
    #store a gray image
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
    faces=model.detectMultiScale(img,1.3,5)
    #Sorting the face with largest bounded box
    faces=sorted(faces,key=lambda f:f[2]*f[3])

    #pick the largest face
    if len(faces)>0:
        f=faces[-1]
        x,y,w,h=f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        #crop and save the largest face
        cropped_face=img[y- offset:y+h + offset,x- offset:x + offset +w]
        cropped_face=cv2.resize(cropped_face,(100,100))
        skip+=1
        if skip % 10==0:
            faceData.append(cropped_face)
            print("Saved so far"+str(len(faceData)))
            cnt+=1
            if cnt==20:
                break


    cv2.imshow("Image Window",img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
#write facedata on the disk
faceData=np.asarray(faceData)
m=faceData.shape[0]
faceData.reshape((m,-1))
print(faceData.shape)   

#save on the disk as np array
filepath=dataset_path + filename +".npy"
np.save(filepath,faceData)
print("data saved successfully"+filepath)

#release camera and destroy all windows
cam.release()
cv2.destroyAllWindows()        
