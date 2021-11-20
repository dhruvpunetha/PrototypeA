import cv2
import numpy as np
def sobel():
    cap = cv2.VideoCapture(0)

    while(1):

        # Take each frame
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([30,150,50])
        upper_red = np.array([255,255,180])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        sobelx = cv2.Sobel(frame,cv2.CV_64F,1,3,ksize=15)
        sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
        cv2.imshow('sobelx',sobelx)
        cv2.imshow('sobely',sobely)
        to_exit = cv2.waitKey(1) & 0xFF==ord('q')
        is_e_pressed=cv2.waitKey(1) & 0xFF==ord('e')
        if is_e_pressed:
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\Filter3.jpg",sobelx)
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\original3.jpg",frame)
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\Filter4.jpg",sobely)
            
        if to_exit:
            break
    cv2.destroyAllWindows()
    cap.release()