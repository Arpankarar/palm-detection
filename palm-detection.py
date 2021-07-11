import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0) # to take video from webcam

mpHands = mp.solutions.hands # before using mp we need to execute this function.
hands=mpHands.Hands()  # it is a function which have 4 parameters. and final values are stored in hands variable.
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0
while True:
    success, img =cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results= hands.process(imgRGB)
    # print(results.multi_hand_landmarks)   to print the corditate of hands in the screen.
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark): # to get the each and every landmark's position
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h) # to change to pixel value from decimals
                print(id,cx,cy)
            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS) # we will use mpDraw to draw the lines between different points.
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),3)  # first 3 for scale and next is for font size
    cv2.imshow("image",img)
    cv2.waitKey(1)