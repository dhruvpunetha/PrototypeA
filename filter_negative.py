import cv2
import numpy as np
def negative():
    cap = cv2.VideoCapture(0)

    while(1):

        
        _, frame = cap.read()
        imgneg=255-frame
        cv2.imshow('negative',imgneg)
        to_exit = cv2.waitKey(1) & 0xFF==ord('q')
        is_e_pressed=cv2.waitKey(1) & 0xFF==ord('e')
        if is_e_pressed:
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\Filter6.jpg",threshold)
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\original6.jpg",frame)
        if to_exit:
            break

    cv2.destroyAllWindows()
    cap.release()