# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:01:40 2019
PRIVACY PROTECTION
@author: AIA
"""
import numpy as np
import argparse
import time
import cv2
import os
os.chdir('G:\VIP CUP\Task-2\Mask\opencvBlog(InceptionV2)\For Github')   #Change according to your directory of the folder

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",type=str,default="images",
help="path to input image")
ap.add_argument("-m", "--mask-rcnn", type=str,default="mask-rcnn-coco",
help="base path to mask-rcnn directory")
ap.add_argument("-v", "--visualize", type=int, default=0,
help="whether or not we are going to visualize each instance")
ap.add_argument("-c", "--confidence", type=float, default=0.45,          #change confidence level if you want
help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.35,
help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["mask_rcnn"],
"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# load our input image and grab its spatial dimensions

# construct a blob from the input image and then perform a forward
# pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
# of the objects in the image along with (2) the pixel-wise segmentation
# for each specific object
def blurring(image):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()
    
    # show timing information and volume information on Mask R-CNN
    print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
    dummyi=np.zeros((H,W),dtype=bool)
    img_copy = image.copy()
    classes=[]
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        classes.append(classID)      #for debugging
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > args["confidence"]:
            # clone our original image so we can draw on it
            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            if(classID==0 or classID==76 or classID==71 or classID==72 or classID==75 or classID==69):
                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH),
                interpolation=cv2.INTER_CUBIC)
                mask = (mask > args["threshold"])
                dummyi[startY:endY, startX:endX]=mask
            else:
                continue
    blur_img=cv2.GaussianBlur(img_copy,(45,45),0)
    img_copy[dummyi]=blur_img[dummyi]
    return img_copy

'''
#For single image
img = cv2.imread('G:/VIP CUP/Task-2/Mask/opencvBlog/mask-rcnn/images/type.jpg')
print(img)
blur=blurring(img)
cv2.imshow("Output", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
############
#For Video #
############

def video_from_dir(dir):
    temp=os.listdir(dir)   #give input directory
    video=[]
    for i in temp:
        if(i.endswith('.MP4')):
            video.append(i)
    return video



if __name__ == '__main__':
    cur_dir=os.getcwd()
    input_path=os.path.join(cur_dir,'nextbatchRun')         #whatever test_directory is named
    video=video_from_dir(input_path)
    for xy in range(len(video)):
        print(video[xy])
        class_id=[]
        #name.append(video[xy])
        capture = cv2.VideoCapture(os.path.join(input_path,video[xy]))
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        codec = cv2.VideoWriter_fourcc(*'MP4V')
        os.chdir(os.path.join(cur_dir,'output'))          #whatever protected saved data folder name is 
        output = cv2.VideoWriter(video[xy], codec, fps, size)
        ret, frame = capture.read()
        while(ret):
            #class_id.append(r['class_ids'])
            img_processed = blurring(frame)
            output.write(img_processed)
            #cv2.imshow('frame',img_processed)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            ret, frame = capture.read()

        capture.release()
        output.release()
        cv2.destroyAllWindows()




