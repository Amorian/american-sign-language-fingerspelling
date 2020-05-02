from PIL import Image
import cv2
import argparse
import math
import os
import numpy as np
import torch
from torchvision import transforms
from model import Net


state_dict = torch.load('model_asl_57.pth')
model = Net()
model.load_state_dict(state_dict)
model.eval()

capture = cv2.VideoCapture(0)
x1 = 220
y1 = 140
x2 = 420
y2 = 340
classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']
label = ''

if __name__ == '__main__':

    # Arguments to OpenCV
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0)
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640)
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480)
    args = parser.parse_args()

    # Video Capture
    capture = cv2.VideoCapture(args.video_source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Size of video to be displayed through OpenCV
    im_width, image_height = (capture.get(3), capture.get(4))

    while True:

        # Read Image
        ret, image_from_cv = capture.read()
        image_from_cv = cv2.flip(image_from_cv, 1)
        image_from_cv = cv2.cvtColor(image_from_cv, cv2.COLOR_BGR2RGB)

        # To quit application press q
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        else:
                # Crop hand from image and process
                roi = image_from_cv[y1:y2, x1:x2]
                #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                #(thresh, roi) = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img = Image.fromarray(roi)
                #img = img.convert('RGB')
                data_transforms = transforms.Compose([
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
                img = data_transforms(img)
                img = img.view(1, img.size(0), img.size(1), img.size(2))
                output = model(img)
                label = classes[torch.argmax(output)]
                
        # Draw bounding box
        cv2.rectangle(image_from_cv, (x1, y1), (x2, y2), (77, 255, 9), 3, 1)

        cv2.putText(image_from_cv, label, (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

        cv2.imshow('ASL Finger Spelling', cv2.cvtColor(
                image_from_cv, cv2.COLOR_RGB2BGR))
                


