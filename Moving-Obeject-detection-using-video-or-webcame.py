import numpy as np
import cv2 as cv
cap = cv.VideoCapture('test2.mp4')


# use this to sustarct bg
algo1=cv.createBackgroundSubtractorMOG2()
algo2=cv.createBackgroundSubtractorKNN(detectShadows=False)

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame,(600,400))
    if frame is None:
        break
    restlt1=algo1.apply(frame)
    result2=algo2.apply(frame)
    
    cnts,hier=cv2.findContours(result2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        epsilon=0.0001*cv2.arcLength(c,True)
        data=cv2.approxPolyDP(c,epsilon,True)
        
        hull=cv2.convexHull(data)
        cv2.drawContours(frame,[c],-1,(50,50,150),2)
        cv2.drawContours(frame,[hull],-1,(0,255,0),2)
    
    cv.imshow('original', frame)
    cv.imshow('MOG2 Result1', restlt1)
    cv.imshow('KNN Result2', result2)
    
    keyboard = cv.waitKey(60)
    if keyboard == 'q' or keyboard == 27:
        break
cap.release()
cv.destroyAllWindows()
