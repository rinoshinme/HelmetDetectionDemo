from time import sleep
import cv2
import argparse
import sys
import numpy as np
import os.path
from glob import glob
#from PIL import image
frame_count = 0             
# used in mainloop  where we're extracting images., and then to drawPred( called by post process)
frame_count_out=0           
# used in post process loop, to get the no of specified class value.
# Initialize the parameters



# Load names of classes

classes = None

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):

    global frame_count
# Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    #print(label)            #testing
    #print(labelSize)        #testing
    #print(baseLine)         #testing

    label_name, label_conf = label.split(':')    #spliting into class & confidance. will compare it with person.
    if label_name == 'Helmet':
                                            #will try to print of label have people.. or can put a counter to find the no of people occurance.
                                        #will try if it satisfy the condition otherwise, we won't print the boxes or leave it.
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        frame_count+=1


    #print(frame_count)
    if(frame_count> 0):
        return frame_count

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out=0
    # classIds = []
    # confidences = []
    # boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []               # have to fins which class have hieghest confidence........=====>>><<<<=======
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                #print(classIds)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person=0 # for counting the classes in this loop.
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
               #this function in  loop is calling drawPred so, try pushing one test counter in parameter , so it can calculate it.
        frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
         #increase test counter till the loop end then print...

        #checking class, if it is a person or not

        my_class='Helmet'                   #======================================== mycode .....
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1
    #if(frame_count_out > 0):
    print(frame_count_out)


    if count_person >= 1:
        path = 'test_out/'
        frame_name=os.path.basename(fn)             # trimm the path and give file name.
        cv.imwrite(str(path)+frame_name, frame)     # writing to folder.
        #print(type(frame))
        cv.imshow('img',frame)
        cv.waitKey(800)


    #cv.imwrite(frame_name, frame)
                                               #======================================mycode.........


class HelmetDetector(object):
    def __init__(self):
        model_configuration = "yolov3-obj.cfg"
        model_weights = "yolov3-obj_2400.weights"

        # load model
        self.net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.outputs_names = self.get_outputs_names(self.net)

        # set parameters
        self.conf_threshold = 0.5  # Confidence threshold
        self.nms_threshold = 0.4   # Non-maximum suppression threshold
        self.input_width = 416       # Width of network's input image
        self.input_height = 416      # Height of network's input image

        class_file = "obj.names"
        self.classes = self.get_class_names(class_file)

    def get_class_names(self, class_file):
        with open(class_file, 'rt') as f:
            class_labels = f.read().rstrip('\n').split('\n')
        return class_labels
    
    @staticmethod
    def get_outputs_names(net):
        # Get the names of all the layers in the network
        layer_names = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def draw_prediction(frame, class_id, conf, left, top, right, bottom):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        conf_text = '%.2f' % conf
        assert class_id < len(self.classes)
        label_text = '%s:%s' % (self.classes[class_id], conf_text)

        label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        label_name, label_conf = label_text.split(':')
        if label_name == 'Helmet':
            cv2.rectangle(frame, (left, top - round(1.5 * label_size[1])), 
            (left + round(1.5 * label_size[1]), top + base_line), 
            (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label_text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    def detect_image(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255, (self.input_width, self.input_height), [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.outputs_names)
        print(outputs)

        # do postprocess
        self.postprocess(image, outputs)

        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    def postprocess(self, image, outputs):
        frame_height = image.shape[0]
        frame_width = image.shape[1]
        class_ids = []
        confidences = []
        boxes = []

        # filter low conf boxes.
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width // 2)
                    top = int(center_y - height // 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # perform non-maximal suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        # draw rectangles.
        count_person = 0
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.draw_prediction(image, class_ids[i], confidences[i], left, top, left + width, right + height)


if __name__ == '__main__':
    model = HelmetDetector()
    img_path =  './images/img.jpg'
    img = cv2.imread(img_path)
    model.detect_image(img)




def main():
    # Process inputs
    winName = 'Deep learning object detection in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    for fn in glob('images/*.jpg'):
        frame = cv.imread(fn)
        frame_count =0

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        #print(t)
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        #print(label)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        #print(label)
