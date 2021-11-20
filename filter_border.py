import cv2 
import numpy as np
def filterline():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while(1):

        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([30,150,50])
        upper_red = np.array([255,255,180])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame,frame, mask=-mask)
        cv2.imshow('Original',frame)
        
        #   cv2.imshow('Orginal',res)
        edges = cv2.Canny(frame,50,200)
        cv2.imshow('Edges',edges)
        to_exit = cv2.waitKey(1) & 0xFF==ord('q')
        is_e_pressed=cv2.waitKey(1) & 0xFF==ord('e')
        if is_e_pressed:
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\Filter1.jpg",edges)
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\original1.jpg",frame)
        if to_exit:
            break

   
    cap.release()
    cv2.destroyAllWindows()