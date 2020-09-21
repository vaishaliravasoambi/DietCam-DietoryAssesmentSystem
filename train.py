from model import FNet
from dataset import *
import torch,os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

def train_model(dataset_path, debug=False, destination_path='', save=False):
	
	learning_rate = 0.001
	num_epochs = 15

	# Check if gpu support is available
	cuda_avail = torch.cuda.is_available()
	print("Cuda available:- ",cuda_avail)

	df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, split=0.90)

	# Data transforms on train set. These are essential as data should be augmented to create varied
	# examples so that our network learns better
	train_transforms = transforms.Compose([
		# transforms.RandomHorizontalFlip(),
		# transforms.RandomVerticalFlip(),
		transforms.ToTensor(),
		# transforms.Resize(400,400),
		transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
	])

	# Tranforms on test data. No other transformation other that normalization is applied.
	# It is necessary to really check our neural network can work without other tranforms
	test_transforms = transforms.Compose([
		transforms.ToTensor(),
		# transforms.Resize(400,400),
		transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
	])
	
	# Loading dataset
	train_ds = ImageDataset(train_df,transform=train_transforms)
	test_ds = ImageDataset(test_df,transform=test_transforms)
	train_len, test_len = len(train_ds), len(test_ds)
	# print(train_len, test_len)

	# creating train and test loaders
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32,shuffle=True,num_workers=4)
	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32,shuffle=True,num_workers=4)

	# Initiate the model
	model = FNet()
	# print(model)

	#if cuda is available, move the model to the GPU
	if cuda_avail:
		torch.cuda.empty_cache()
		model.cuda()

	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)   # optimize all cnn parameters
	loss_func = nn.CrossEntropyLoss()	# the target label is not one-hotted

	# Function to evaluate test data
	def test():

		model.eval()	# Change model mode to evaluation

		test_acc = 0.0	# total test accuracy
		test_loss = 0.0	# total test loss
		test_running_loss = 0.0	# test batch loss
		test_running_acc = 0.0	# test batch accuracy
		test_label_ct = 0.0	# test batch label count
		test_label = 0.0	# test complete label count
		for i, (images, labels) in enumerate(test_loader):

			if cuda_avail:
				# print("IN CUDA....")
				images = images.cuda()
				labels = labels.cuda()

			# Predict classes using images from the test set
			outputs = model(images)	# forward pass
			t_loss = loss_func(outputs, labels)	# calculate loss
			test_current_loss = t_loss.cpu().item() * images.size(0)

			test_running_loss += test_current_loss
			test_loss += test_current_loss

			_, prediction = torch.max(outputs.data, 1)
			test_current_acc = torch.sum(prediction == labels.data).item() * 1.0
			test_label_ct += labels.size(0)
			test_label += labels.size(0)
			test_running_acc += test_current_acc
			test_acc += test_current_acc

			if debug == True: # Print test loss and accturacy after 10 minibatches
				if i%10 == 9:
					print("[%d] test loss: %f test acc: %f" %(i+1, test_running_loss/test_label_ct, test_running_acc*100/test_label_ct))
					test_running_loss = 0.0
					test_running_acc = 0.0
					test_label_ct = 0.0

		test_acc = test_acc / test_label
		test_loss = test_loss / test_label
		return test_acc, test_loss

	for epoch in range(num_epochs):

		model.train()	# Change model mode to train

		train_acc = 0.0	# total train accuracy
		train_loss = 0.0 # total train loss
		running_loss = 0.0	# train batch loss
		running_acc = 0.0	# train batch accuracy
		label_ct = 0.0	# batch label count
		train_label = 0.0	# total label count

		for i, (images, labels) in enumerate(train_loader):
			# Move images and labels to gpu if available
			if cuda_avail:
				images = images.cuda()
				labels = labels.cuda()

			# Clear all accumulated gradients
			optimizer.zero_grad()

			# Predict classes using images from the train set
			outputs = model(images)

			# Compute the loss based on the predictions and actual labels
			loss = loss_func(outputs, labels)

			# Backpropagate the loss
			loss.backward()

			# Adjust parameters according to the computed gradients
			optimizer.step()

			current_loss = loss.cpu().item() * images.size(0)
			running_loss += current_loss
			train_loss += current_loss

			_, prediction = torch.max(outputs, 1)
			current_acc = torch.sum(prediction == labels).item() * 1.0 
			label_ct += labels.size(0)
			train_label += labels.size(0)
			running_acc += current_acc 
			train_acc += current_acc
			if debug == True: # Print train loss and accturacy after 30 minibatches
				if i%30 == 29:
					print("[%d %d] train loss: %f train acc: %f" %(epoch+1, i+1, running_loss/label_ct, (running_acc/label_ct)*100))
					running_loss = 0.0
					running_acc = 0.0
					label_ct = 0.0

		# Compute the average acc and loss over all training images
		train_acc = train_acc / train_label  
		train_loss = train_loss / train_label

		# Evaluate on the test set
		test_acc, test_loss = test()

		# Print the metrics
		print("Epoch %d ,Train Accuracy: %f ,Test Accuracy: %f, TrainLoss: %.4f ,TestLoss: %.4f \n" %(epoch+1, train_acc*100, test_acc*100, train_loss, test_loss))


		if save == True:	# Saving the model
			path = os.path.abspath(destination_path)+'/model_'+str(epoch+1)+'.pt'
			# print(path)
			torch.save(model.state_dict(), path)
			print("Model saved...")

	# save loss and accuracy of train and test as float tensors
	loss = torch.FloatTensor([train_loss, test_loss])
	accuracy = torch.FloatTensor([train_acc, test_acc])

	return loss, accuracy


if __name__ == '__main__':
	loss, accuracy =  train_model('./Data/vegies_n_fruits', save=True, destination_path='./',debug=True)