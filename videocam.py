import cv2

cam=cv2.VideoCapture(0)

#read image from webcam
while True:
    success,img=cam.read()
    if not success:
        print("reading failure")
    cv2.imshow("Image window",img)
    cv2.waitKey(1)    