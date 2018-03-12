import sys
import os
import pickle
import numpy as np
from skimage import data, exposure
from skimage.color import rgb2grey
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from data import unpickle_data
from utils.visualization import plot_confusion_matrix

class ShallowHogModel():
	
	def __init__(self, model_type, debug_level):
	
		self.model_type = model_type
		self.num_histogram_orientations = 9
		self.num_pixels_per_cell = 8
		self.num_cells_per_block = 2
		self.block_norm = 'L2-Hys'
		self.kernel_type = 'rbf'
		self.scoring_function = 'accuracy'
		self.num_cv_folds = 5
		self.num_parallel_jobs = 3
		self.random_state = 12345
		self.debug_level = debug_level
		
	def compute_descriptor(self, image, visualise=False):
	
		if not visualise:
			descriptor = hog(image,orientations=self.num_histogram_orientations,
				pixels_per_cell=(self.num_pixels_per_cell,self.num_pixels_per_cell),
				cells_per_block=(self.num_cells_per_block,self.num_cells_per_block),
				block_norm=self.block_norm,visualise=visualise,transform_sqrt=True,feature_vector=True)
		else:
			descriptor, hog_image = hog(image,orientations=self.num_histogram_orientations,
				pixels_per_cell=(self.num_pixels_per_cell,self.num_pixels_per_cell),
				cells_per_block=(self.num_cells_per_block,self.num_cells_per_block),
				block_norm=self.block_norm,visualise=visualise,transform_sqrt=True,feature_vector=True)

			fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4),sharex=True,sharey=True)

			ax1.axis('off')
			ax1.imshow(image,cmap=plt.cm.gray)
			ax1.set_title('Input image')

			# Rescaling histogram for better display
			rescaled_hog_image = exposure.rescale_intensity(hog_image,in_range=(0,10))

			ax2.axis('off')
			ax2.imshow(rescaled_hog_image,cmap=plt.cm.gray)
			ax2.set_title('Histogram of Oriented Gradients')
			
			plt.show()
		
		return descriptor

	def prepare_batch(self, batch):
	
		batch_images = batch['data']
		batch_labels = batch['labels']
		
		# Determining batch size and image size
		batch_size = batch_images.shape[0]
		image_size = int(np.sqrt(batch_images.shape[1]/3))

		# Allocating storage for stacked image descriptors
		num_block_locations_along_axis = int(image_size/self.num_pixels_per_cell-self.num_cells_per_block+1)
		num_block_locations = num_block_locations_along_axis*num_block_locations_along_axis
		descriptor_size = num_block_locations*self.num_cells_per_block*self.num_cells_per_block*self.num_histogram_orientations
		batch_descriptors = np.zeros((batch_size,descriptor_size))

		# Iterating over rows to transform them into images and computing descriptors
		for i in range(batch_size):
		
			# Rearranging data into an RGB image
			image = batch_images[i,:].reshape(3,image_size,image_size).transpose([1,2,0])

			# conversion to gray-scale image
			grayscale_image = rgb2grey(image)

			# Computing and stacking HOG descriptor
			if self.debug_level > 2:
				self.compute_descriptor(grayscale_image,visualise=True)
				exit(1)
			else:
				descriptor = self.compute_descriptor(grayscale_image)
			
			batch_descriptors[i,:] = descriptor
		
			if self.debug_level > 0:
				print('Processed {} of {} images'.format(i,batch_size))

		return batch_descriptors, batch_labels

	def prepare_data(self, input_data_path):
		
		print('Preparing data')
		
		all_training_descriptors, all_test_descriptors = None, None
		all_training_labels, all_test_labels = None, None
		
		for path in os.listdir(input_data_path):
		
			item_path = os.path.join(input_data_path,path)
			
			if os.path.isfile(item_path) and (path.startswith('sampled_data_batch') or path.startswith('sampled_test_batch')):
				batch = unpickle_data(item_path)

				# Computing descriptors from this batch
				descriptors, labels = self.prepare_batch(batch)
				
				if path.startswith('sampled_data_batch'):
					if all_training_descriptors is None:
						all_training_descriptors = descriptors
						all_training_labels = labels
					else:
						all_training_descriptors = np.append(all_training_descriptors,descriptors,axis=0)
						all_training_labels = np.append(all_training_labels,labels)
				elif path.startswith('sampled_test_batch'):
					if all_test_descriptors is None:
						all_test_descriptors = descriptors
						all_test_labels = labels
					else:
						all_test_descriptors = np.append(all_test_descriptors,descriptors,axis=0)
						all_test_labels = np.append(all_test_labels,labels)

				if self.debug_level > 0:
					print('Prepared data from batch {}'.format(path))

		return all_training_descriptors, all_training_labels, all_test_descriptors, all_test_labels

	def train(self, training_data, training_labels, output_data_path):
	
		# Building model
		print('Building model')
		
		if self.model_type == 'svm':
			model = SVC(kernel=self.kernel_type,degree=2,probability=True,
				tol=1e-4,max_iter=1000,random_state=self.random_state)

			if self.kernel_type == 'linear':
				grid = GridSearchCV(model,
					param_grid=dict(
						C=[x for x in np.arange(0.0001,0.001,0.0001)]),
					scoring=self.scoring_function,
					cv=self.num_cv_folds,
					n_jobs=self.num_parallel_jobs,
					verbose=self.debug_level)
			elif self.kernel_type in ['rbf','poly']:
				grid = GridSearchCV(model,
					param_grid=dict(
						C=[10.0,100.0,1000.0],				# best 10.0
						gamma=[0.0001,0.001,0.01]),			# best 0.001
					scoring=self.scoring_function,
					cv=self.num_cv_folds,
					n_jobs=self.num_parallel_jobs,
					verbose=self.debug_level)
		elif self.model_type == 'gradient_boosting':
			model = GradientBoostingClassifier(random_state=self.random_state)
			
			grid = GridSearchCV(model,
				param_grid=dict(
					learning_rate=[0.01],
					n_estimators=[500],
					max_depth=[10],
					max_features=['sqrt'],
					subsample=[0.75],
					min_samples_split=[200],
					min_samples_leaf=[100]),
				scoring=self.scoring_function,
				cv=self.num_cv_folds,
				n_jobs=self.num_parallel_jobs,
				verbose=self.debug_level)
		else:
			print('Unsupported model type. Quitting')
			
			exit(2)
		
		grid.fit(training_data,training_labels)

		print('Best accuracy of {} found for parameters: {}'.format(grid.best_score_,grid.best_params_))

		# Obtaining the final model
		best_model = grid.best_estimator_
		
		# Serializing final model
		print('Serializing model')
		
		model_file_name = os.path.join(output_data_path,'hog_{}_model.pkl'.format(self.model_type))
		
		with open(model_file_name,'wb') as f:
			pickle.dump(best_model,f)
		
		return best_model

	def evaluate(self, model, test_data, test_labels, metadata_path, output_data_path):
	
		# Reading class names
		metadata = unpickle_data(metadata_path)
		unique_class_names = [l.decode('ascii') for l in metadata[b'label_names']]
		num_classes = len(unique_class_names)
		
		# Predicting test labels
		predicted_labels = model.predict(test_data)
		
		# Plotting confusion matrix
		conf_matrix = confusion_matrix(test_labels,
			predicted_labels,
			labels=[i for i in range(num_classes)]
		)
	
		plot_confusion_matrix(conf_matrix,
			unique_class_names,
			num_classes,
			10,10,
			output_data_path,
			'confusion_matrix'
		)
