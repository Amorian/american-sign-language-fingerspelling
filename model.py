import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 26

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
		self.drop = nn.Dropout2d()
		self.fc1 = nn.Linear(256, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, nclasses)
		
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = self.drop(F.max_pool2d(F.relu(self.conv3(x)), 2))
		x = x.view(-1, 256)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.dropout(x, training=self.training)
		x = self.fc3(x)
		return F.log_softmax(x, dim=-1)
