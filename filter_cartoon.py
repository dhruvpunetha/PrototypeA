import numpy as np
import scipy
from scipy import stats
import cv2  #open source software with pre built functons for image processing
from collections import defaultdict #defalt dictionary is sub class of dictionary
# we prefer default dict as it will not raise a key error


#updating value of centroid
def up_coun(C,hist):   
    while True:
        groups=defaultdict(list)

        for i in range(len(hist)):
            if(hist[i] == 0):
                continue
            d=np.abs(C-i)
            index=np.argmin(d)
            groups[index].append(i)

        new_centroid=np.array(C)
        for i,indice in groups.items():
            if(np.sum(hist[indice])==0):
                continue
            new_centroid[i]=int(np.sum(indice*hist[indice])/np.sum(hist[indice]))

        if(np.sum(new_centroid-C)==0):
            break
        C=new_centroid

    return C,groups

# Instead of using heavy models like gan we are simply using k means clustering
#k means partations n observations to k clusters
# we do this in cartoonization to basically reduce the shades and a variety of shades will have same shades

def K_histogram(hist):

    alpha=0.001
    N=80
    C=np.array([128])

    while True:
        C,groups=up_coun(C,hist)

        new_centroid=set()
        for i,indice in groups.items():
            if(len(indice)<N):
                new_centroid.add(C[i])
                continue

            z, pval=stats.normaltest(hist[indice])
            if(pval<alpha):
                left=0 if i==0 else C[i-1]
                right=len(hist)-1 if i ==len(C)-1 else C[i+1]
                delta=right-left
                if(delta >=3):
                    c1=(C[i]+left)/2
                    c2=(C[i]+right)/2
                    new_centroid.add(c1)
                    new_centroid.add(c2)
                else:
                    new_centroid.add(C[i])
            else:
                new_centroid.add(C[i])
        if(len(new_centroid)==len(C)):
            break
        else:
            C=np.array(sorted(new_centroid))
    return C

# The main  function to cartoonize image
def cartooniz(img):

    kernel=np.ones((2,2), np.uint8)
    output=np.array(img)
    x,y,c=output.shape
    for i in range(c):
        output[:,:,i]=cv2.bilateralFilter(output[:,:,i],5,150,150)
        #output[:,:,i]=cv2.medianBlur(output[:,:,i],5)
        #output[:,:,i]=cv2.GaussianBlur(output[:,:,i],(5,5),150)
#bilateral filter to smoothen out the image it replaces each pixal with weighted average of values nearby.
#we didnt use gaussian or median blur they smoothen the edges
    edge=cv2.Canny(output, 100, 200)
    #canny is used to find the edge of the image
    #canny edge just looks for local maxima and uses it otherwise it supresses the image
   
    output=cv2.cvtColor(output,cv2.COLOR_RGB2HSV)

    hists = []
    hist,_=np.histogram(output[:,:,0],bins =np.arange(180+1))
    hists.append(hist)
    hist,_=np.histogram(output[:,:,1],bins =np.arange(256+1))
    hists.append(hist)
    hist,_=np.histogram(output[:,:,2],bins =np.arange(256+1))
    hists.append(hist)
    '''hist,_=np.histogram(output[:,:,0],bins =np.arange(180))
    hists.append(hist)
    hist,_=np.histogram(output[:,:,1],bins =np.arange(256))
    hists.append(hist)
    hist,_=np.histogram(output[:,:,2],bins =np.arange(256))
    hists.append(hist)'''


    C=[]
    for h in hists:
        C.append(K_histogram(h))
    #print("centroids: {0}".format(C))

    output=output.reshape((-1,c))
    for i in range(c):
        channel=output[:,i]
        index=np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:,i]=C[i][index]
    output=output.reshape((x,y,c))
    output=cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
#unlike edges counturs joins the edges
    contours,_=cv2.findContours(edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output,contours,-1,0,thickness=1)
    #cartoon = cv2.bitwise_and(output, output, mask=contours)
    
    #here we are eroding out image using erode that will thicken the outer lines
    for i in range(3):
        output[:,:,i]=cv2.erode(output[:,:,i], kernel, iterations=1)
    #Laplacian = cv2.Laplacian(output,cv2.CV_8U, ksize=11)
    #output=output-Laplacian
    return output

#output=caart(cv2.imread("original.jpg"))
#cv2.imwrite("cartoon.jpg", output)

#we applied erosion k means ,contours,canny bilateral filter

def cartooni():
    videoCaptureObject = cv2.VideoCapture(0)

    "out = cv2.VideoWriter('out.mp3', cv2.VideoWriter_fourcc(*'MP4V'), 24, (720, 1280))"
    while(True):
        ret,img = videoCaptureObject.read()
        imgct=cartooniz(img)
        cv2.imshow("cartoonized",imgct)
        
        cv2.imshow("original",img)
        to_exit = cv2.waitKey(1) & 0xFF==ord('q')
        is_e_pressed=cv2.waitKey(1) & 0xFF==ord('e')
        if is_e_pressed:
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\Filter5.jpg",imgct)
            cv2.imwrite("D:\Sem5Mini\PrototypeA\output\original2.jpg",img)
        if to_exit:
            break
    videoCaptureObject.release()
    cv2.destroyAllWindows()