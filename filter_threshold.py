import cv2
import numpy as np
def filterheat():
    cap = cv2.VideoCapture(0)

    while(1):

        # Take each frame
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        retval, threshold = cv2.threshold(frame, 100, 155, cv2.THRESH_BINARY)
        cv2.imshow('thodl',threshold)
        to_exit = cv2.waitKey(1) & 0xFF==ord('q')
        is_e_pressed=cv2.waitKey(1) & 0xFF==ord('e')
        if is_e_pressed:
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\Filter6.jpg",threshold)
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\original6.jpg",frame)
        if to_exit:
            break

    cv2.destroyAllWindows()
    cap.release()