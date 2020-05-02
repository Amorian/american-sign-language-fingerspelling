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

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']

for i in classes:
    path = 'sample_predict/' + i + '.jpeg'
    img = cv2.imread(path)
    img = Image.fromarray(img)
    data_transforms = transforms.Compose([
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    img = data_transforms(img)
    img = img.view(1, img.size(0), img.size(1), img.size(2))
    output = model(img)
    label = classes[torch.argmax(output)]
    print(i, " ", label)


