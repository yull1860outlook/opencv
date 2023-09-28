import cv2
import numpy as np
from matplotlib import pyplot as plt

# OpenCV sample code:
#https://github.com/rchavezj/OpenCV_Projects/tree/master/Sec02_Image_Manipulations/02_Rotations

input = cv2.imread('error.jpg')
img = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.title('my picture')

def grayscale():
    grey = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    row,col = 2,1
    fig,axs = plt.subplots(row,col,figsize=(5,5))
    fig.tight_layout()

    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(grey,cmap='gray'  )
    axs[1].set_title('Grey Image')

    plt.show()

###HSV space
def HSV_space():
    hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    row,col = 2,2
    fig,axs = plt.subplots(row,col,figsize=(5,5))
    fig.tight_layout()
    axs[0,0].imshow(hsv)
    axs[0,0].set_title('HSV Image')

    axs[0,1].imshow(cv2.cvtColor(hsv[:,:,0],cv2.COLOR_BGR2RGB))
    axs[0,1].set_title('Hue')
    axs[1,0].imshow(cv2.cvtColor(hsv[:,:,1],cv2.COLOR_BGR2RGB))
    axs[1,0].set_title('Saturation')
    axs[1,1].imshow(cv2.cvtColor(hsv[:,:,2],cv2.COLOR_BGR2RGB))
    axs[1,1].set_title('Value')
    plt.show()


def amplify_single_plane():
    B,G,R = cv2.split(input)
    row,col = 2,1
    fig,axs = plt.subplots(row,col,figsize=(5,5))
    fig.tight_layout()

    merge = cv2.merge([B,G,R])
    axs[0].imshow(cv2.cvtColor(merge,cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    amplified = cv2.merge([B,G,R+100])
    axs[1].imshow(cv2.cvtColor(amplified ,cv2.COLOR_BGR2RGB) )
    axs[1].set_title('merged with red Amplified')

    plt.show()

def histogram():
    color = ('b','g','r')
    for i,col in enumerate(color):
        histogram2 = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histogram2,color = col)
        plt.xlim([0,256])
    #histogram2 = cv2.calcHist([image],[0],None,[256],[0,256])    
    #plt.hist(img.ravel(),256,[0,256])
    plt.show()

def drawingOnImage():
    cv2.line(img,(0,0),(150,150),(255,255,255),15)
    cv2.rectangle(img,(15,25),(200,150),(0,255,0),5)
    cv2.circle(img,(10,10),3,(0,0,255),2)

    pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
    cv2.polylines(img,[pts],True,(0,255,255),3)

    cv2.putText(img,'text to show',(0,130),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)


if __name__ == '__main__':
    #grayscale()
    #HSV_space()
    #amplify_single_plane()
    histogram()