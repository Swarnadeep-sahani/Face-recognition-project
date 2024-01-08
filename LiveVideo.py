import cv2

#create camera object
cam=cv2.VideoCapture(0);

#model
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#read image from camera
while True:
    success,img=cam.read()
    if not success:
        print("reading camera failure")
    faces=model.detectMultiScale(img,1.3,5)

    for f in faces:
        x,y,w,h=f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Image Window",img)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    #release camera and destroy all windows
cam.release()
cv2.destroyAllWindows()        
