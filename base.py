import filter_border
import filter_laplace
import filter_sobel
import filter_cartoon_dilated
import filter_cartoon
import filter_threshold
print("Hello Please choose a filter")
ch=int(input())
if(ch==1):
    filter_border.filterline()
if(ch==2):    
    filter_laplace.laplace()
if(ch==3):    
    filter_sobel.sobel()
if(ch==4):    
    filter_cartoon.cartooni()
if(ch==5):    
    filter_cartoon_dilated.cartooni()
if(ch==6):    
    filter_threshold.filterheat()