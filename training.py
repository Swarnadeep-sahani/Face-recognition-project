import cv2
import numpy as np
import os

#data
dataset_path="./data/"
faceData=[]
labels=[]
nameMap={}

classId=0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId]=f[:-4]
        #x-value
        dataItem=np.load(dataset_path+f)
        m=dataItem.shape[0]
        faceData.append(dataItem)

        #y-data
        target=classId*np.ones((m,))
        classId+=1
        labels.append(target)

xT=np.concatenate(faceData,axis=0)
yT=np.concatenate(labels,axis=0).reshape((-1,1))

print(xT.shape)
print(yT.shape)
# print(nameMap)

#Algorithm
def dist(p,q):
    q = q.reshape(p.shape)
    return np.sqrt(np.sum((p-q)**2))

def knn(X,y,xt,k=5):
    m=X.shape[0]
    dlist=[]

    for i in range(m):
        d=dist(X[i],xt)
        dlist.append((d,y[i]))
    dlist=sorted(dlist)
    # dlist=np.array(dlist[:k])
    dlist = np.array(dlist[:k], dtype=object)
    labels=dlist[:,1]

    labels,cnts=np.unique(labels,return_counts=True)
    idx=cnts.argmax()
    pred=labels[idx]
    return int(pred)

#prediction part-3
cam=cv2.VideoCapture(0);
dataset_path="./data/"
offset=20
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    success,img=cam.read()
    if not success:
        print("reading camera failure")    
    faces=model.detectMultiScale(img,1.3,5)

    for f in faces:
        x,y,w,h=f

        #crop and save the largest face
        cropped_face=img[y- offset:y+h + offset,x- offset:x + offset +w]
        cropped_face=cv2.resize(cropped_face,(100,100))
        #cv2.imshow("Image Window",img)
        #prediction the name using KNN
        classPrediction=knn(xT,yT,cropped_face.flatten())
        #name 
        namePrediction=nameMap[classPrediction]
        #display name and box
        cv2.putText(img, namePrediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Example font face and size

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Prediction window",img)    
    key=cv2.waitKey(1)
    if key==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


        
