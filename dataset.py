# import required libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os,csv
from PIL import Image

def create_meta_csv(dataset_path, destination_path):
	
	# Change dataset path accordingly
	DATASET_PATH = os.path.abspath(dataset_path)
	print(DATASET_PATH)

	classes = {'Apple':0, 'Banana': 1, 'Carrot':2, 'Cucumber': 3, 'Onion':4, 'Orange': 5, 'Tomato':6}
	if not os.path.exists(os.path.join(DATASET_PATH, "/dataset_attr.csv")):
		if destination_path == None:
			destination_path = dataset_path

		# Change destination path accoridingly
		DEST_PATH = os.path.abspath(destination_path)
		print(DEST_PATH)
		path = DEST_PATH + "/dataset_attr.csv"

		if not os.path.exists(path):
			print("file does not exists writing....")
			# write out as dataset_attr.csv in destination_path directory
			with open(path,'w') as f:	
				wr = csv.writer(f)
				pth = DEST_PATH.split(os.sep)[-1]
				# print("PATH: ",pth)
				
				flag = True
				if(pth != 'vegies_n_fruits' ): # work around for path setting for running th' main and test file
					print("path,label",file=f)
					flag = False

				# Iterate through directories and fies
				for subdir, dirs, files in os.walk(DATASET_PATH):
					print("Subdir: ",subdir)
					for file in files:
						lis = []
						subd = subdir.split(os.sep)[-1]
						# print(subd)
						if(subd == 'vegies_n_fruits') and flag: # work
							lis.append('path')
							lis.append('label')
						else:

							lis.append(os.path.join(subdir, file))
							lis.append(classes[subd])
						print(lis)
						wr.writerow(lis)
			f.close()
		else:
			print("file exists no writing....")
		# if no error
		return True

def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
	if create_meta_csv(dataset_path, destination_path=destination_path):
		dframe = pd.read_csv(os.path.join(destination_path, 'dataset_attr.csv'))
		print("CSV CREATED")


	# shuffle if randomize is True or if split specified and randomize is not specified
	# so default behavior is split
	if randomize == True or (split != None and randomize == None):
		# shuffle the dataframe here
		dframe = dframe.sample(frac=1).reset_index(drop=True)

	if split != None:
		train_set, test_set = train_test_split(dframe, split)
		return dframe, train_set, test_set

	return dframe

def train_test_split(dframe, split_ratio):
	# divide into train and test dataframes
	train_data = dframe.sample(frac=split_ratio).reset_index(drop=True)
	test_data = dframe.drop(train_data.index).reset_index(drop=True)
	return train_data, test_data

class ImageDataset(Dataset):
	
	def __init__(self, data, transform=None):
		self.data = data
		self.transform = transform
		self.classes = data['label'].unique()# get unique classes from data dataframe
		# print(self.classes) # Uncomment to print unique classes

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path = self.data.iloc[idx]['path']
		image = Image.open(img_path) # load PIL image
		image = image.resize((400,400))
		label = self.data.iloc[idx]['label'] # get label (derived from self.classes; type: int/long) of image
		# image.show()   # Uncomment to see the image
		if self.transform:
			image = self.transform(image)

		return image, label


if __name__ == "__main__":
	# test config
	dataset_path = './Data/vegies_n_fruits'
	dest = './Data/'
	classes = 7
	# total_rows = 4323
	randomize = True
	clear = True

	# test_create_meta_csv()
	df, trn_df, tst_df = create_and_load_meta_csv_df(dataset_path, destination_path=dest, randomize=randomize, split=0.99)
	print(df.describe())
	print(trn_df.describe())
	print(tst_df.describe())
	train_dataset = ImageDataset(trn_df)

	print("LEN:-",len(train_dataset))
	train_dataset.__getitem__(4)
