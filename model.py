# Imports
import torch.nn as nn
import torch.nn.functional as F

class FNet(nn.Module):
	
	# Initializes CNN
	def __init__(self):

		
		super(FNet, self).__init__()	# Calling base class constructor
		
		
		self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
		self.bn3 = nn.BatchNorm2d(128)

		# Fully-Connected layers transform output of conv. layers to final output
		self.dropout_rate = 0.2
		self.fc1 = nn.Linear(in_features = 128*50*50, out_features = 128)
		self.fc2 = nn.Linear(in_features = 128, out_features = 64)
		self.fc3 = nn.Linear(in_features = 64, out_features = 7)

	def forward(self, x):
		# forward propagation

		# we apply the convolution layers, followed by batch normalisation

		x = self.bn1(self.conv1(x))  
		x = F.relu(F.max_pool2d(x, 2)) 
		x = self.bn2(self.conv2(x)) 
		x = F.relu(F.max_pool2d(x, 2)) 
		x = self.bn3(self.conv3(x)) 
		x = F.relu(F.max_pool2d(x, 2))
		# print(x.shape)
		# flatten the output for each image
		x = x.view(-1,128*50*50)  

		# apply 2 fully connected layers with dropout
		x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_rate) # batch_size x 128
		x = F.relu(self.fc2(x)) # batch * 5
		x = self.fc3(x)

		return F.log_softmax(x, dim=1)