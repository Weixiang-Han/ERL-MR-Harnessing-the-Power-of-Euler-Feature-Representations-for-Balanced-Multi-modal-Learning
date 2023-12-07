import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random

MEAN_OF_ACC = 0.30706118047445236
STD_OF_ACC = 0.5935681217245586
MEAN_OF_GYRO = -0.31099145640546716
STD_OF_GYRO = 44.133894889355076
random.seed(42)






class Multimodal_imdataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x1, x2, y,reduced_labels):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		self.labels = y.tolist() #tolist

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()
		self.reduced_labels=reduced_labels

		self.label_to_index = {}
		for i, label in enumerate(self.reduced_labels):
			if label not in self.label_to_index:
				self.label_to_index[label] = []
			self.label_to_index[label].append(i)


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		activity_label = self.labels[idx]
		# 直接按索引采样 y
		# 从标签索引字典中随机选择一个与 y 相同的索引
		label_indexes = self.label_to_index[int(activity_label.item())]
		x1_index = np.random.choice(label_indexes)
		sensor_data1 = self.data1[x1_index]
		sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		sensor_data2 = torch.unsqueeze(sensor_data2, 0)


		return sensor_data1, sensor_data2, activity_label


class Multimodal_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x1, x2, y):

		self.data1 = x1.tolist() #concate and tolist
		self.data2 = x2.tolist() #concate and tolist
		self.labels = y.tolist() #tolist

		self.data1 = torch.tensor(self.data1) # to tensor
		self.data2 = torch.tensor(self.data2) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data1 = self.data1[idx]
		sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		sensor_data2 = torch.unsqueeze(sensor_data2, 0)

		activity_label = self.labels[idx]

		return sensor_data1, sensor_data2, activity_label


class Singlemodal_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x, y):

		self.data = x.tolist() #concate and tolist
		self.labels = y.tolist() #tolist

		self.data = torch.tensor(self.data) # to tensor
		self.labels = torch.tensor(self.labels)
		self.labels = (self.labels).long()


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data = self.data[idx]
		sensor_data = torch.unsqueeze(sensor_data, 0)

		activity_label = self.labels[idx]

		return sensor_data, activity_label
		


