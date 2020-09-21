import torch, torchvision, os
from torchvision import transforms
import torch
from PIL import Image
import argparse
from model import FNet

test_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

classes = ['Apple', 'Banana', 'Carrot', 'Cucumber', 'Onion', 'Orange', 'Tomato']


def process(image_path):
	img = Image.open(image_path)
	img = img.resize((400,400))
	img = test_transforms(img)
	# Add an extra dimension to image tensor representing batch size
	img = img.unsqueeze_(0)

	return img.to(device)

def eval(image_path, trained_model_path):

	# image_path = os.path.abspath(image_path)
	# model_path = os.path.abspath(trained_model_path)
	
	# print(device)
	model_ = torch.load(trained_model_path)
	model_.to(device)

	# Set model in evaluation mode
	model_.eval()
	
	image = process(image_path)

	# image_tensor.to(device)
	outputs = model_(image)
	_,preds = torch.max(outputs,1)


	return classes[preds.item()]
	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--img' ,help="Image Path")
	parser.add_argument('--model' ,help="Model path")

	args=parser.parse_args()


	image_path = os.path.abspath(args.img)
	model_path = os.path.abspath(args.model)

	prediction = eval(image_path, model_path)
	print("Predicted = ",prediction)




