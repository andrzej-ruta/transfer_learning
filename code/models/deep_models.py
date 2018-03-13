
import os
import pickle
from urllib.request import ProxyHandler, build_opener, install_opener
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from skimage.transform import resize
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Flatten
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocess_input
import matplotlib.pyplot as plt

from data import unpickle_data
from utils.visualization import plot_confusion_matrix

class DeepTransferModel():
	
	def __init__(self, model_architecture, debug_level, proxy_url=None, proxy_user_name=None, proxy_password=None):

		self.model_architecture = model_architecture
		
		self.tsne_perplexity = 30.0
		self.tsne_early_exaggeration = 20.0
		self.tsne_learning_rate = 120.0
		
		self.min_explained_variance_ratio = 0.95
		
		self.kernel_type = 'rbf'
		self.scoring_function = 'accuracy'
		self.num_cv_folds = 5
		self.num_parallel_jobs = 3
		self.random_state = 12345
		self.debug_level = debug_level

		if proxy_url:
		
			# Setting up proxy
			proxy = ProxyHandler({'https': '{}:{}@{}'.format(proxy_user_name,proxy_password,proxy_url)})
			opener = build_opener(proxy)
			install_opener(opener)
		
		# Downloading pre-trained weights
		if model_architecture == 'vgg16':
			self.original_image_size = (224,224)
			model = VGG16(include_top=False,
				weights='imagenet',
				pooling=None)
		elif model_architecture == 'inception_v3':
			self.original_image_size = (299,299)
			model = InceptionV3(include_top=False,
				weights='imagenet',
				pooling=None)
		else:
			print('Unsupported model type. Quitting')
			
			exit(2)
		
		# Plugging in the transfer model architecture to the final architecture
		input = Input(shape=(self.original_image_size[0],self.original_image_size[1],3),
			name = 'image_input')
		transfer_model_output = model(input)
		transfer_model_output = Flatten(name='flatten')(transfer_model_output)
		self.transfer_model = Model(input,transfer_model_output)
		
	def visualize_image_representation(self, input_data_path, metadata_path, output_data_path,
		dim_reduction_method, num_examples_per_class):

		# Initializing the dimensionality reduction model
		if dim_reduction_method == 'pca':
			dim_reduction_model = PCA(n_components=2)
		elif dim_reduction_method == 't-sne':
			dim_reduction_model = TSNE(perplexity=self.tsne_perplexity,
				early_exaggeration = self.tsne_early_exaggeration,
				learning_rate = self.tsne_learning_rate,
				random_state=self.random_state)
		else:
			print('Unsupported dimensionality reduction method. Quitting')
			
			exit(2)

		# Opening data file
		dataset = unpickle_data(input_data_path)
		labels = np.array(dataset['labels'])
		images = dataset['data']
		
		# Determining image size
		image_size = int(np.sqrt(images.shape[1]/3))

		# Reading class names
		metadata = unpickle_data(metadata_path)
		unique_class_names = [l.decode('ascii') for l in metadata[b'label_names']]
		num_classes = len(unique_class_names)

		# Building random sample for visualization
		total_num_images = num_classes*num_examples_per_class
		all_features = np.zeros((total_num_images,self.transfer_model.output_shape[1]))
		all_labels = np.empty(total_num_images)
		image_number = 1

		for class_id in range(num_classes):
			class_indices = labels == class_id
			class_images = images[class_indices,:]
			total_num_class_images = class_images.shape[0]
			selected_indices = np.random.choice(total_num_class_images,
				num_examples_per_class,replace=False)
			sample_class_images = class_images[selected_indices,:]
			
			for i in range(num_examples_per_class):
			
				# Extracting image
				image = sample_class_images[i,:].reshape(3,image_size,image_size).transpose([1,2,0])
				
				# Upsampling image to original size used while training the net
				resized_image = resize(image,self.original_image_size,preserve_range=True)
				
				# Reshaping the image data to fit the net
				net_input = np.expand_dims(resized_image,axis=0)
				
				# Preprocessing that replicates the one using at the training stage
				# (mean RGB value extraction only)
				net_input = vgg_preprocess_input(net_input)
				
				# Extracting feature vector from the last network layer
				features = self.transfer_model.predict(net_input)[0]
				
				all_features[image_number-1,:] = features
				all_labels[image_number-1] = class_id
				
				if self.debug_level > 0:
					print('Processed {} of {} images'.format(image_number,total_num_images))
				
				image_number += 1

		# Performing dimensionality reduction
		vis_data = dim_reduction_model.fit_transform(all_features)

		# Visualization of low dimensional embedding
		cmap = plt.get_cmap('spectral',10)
		fig, ax = plt.subplots()
		
		for class_id in range(num_classes):
			class_vis_data = vis_data[all_labels == class_id,:]
			ax.scatter(class_vis_data[:,0],class_vis_data[:,1],s=5,c=cmap(class_id),
				label=unique_class_names[class_id],cmap=cmap)

		plt.legend(loc=1,bbox_to_anchor=(1.3,1.0),borderaxespad=0.)
		plt.title('Low-dimensional data embedding ({})'.format(dim_reduction_method),fontsize=14)
		plt.savefig(os.path.join(output_data_path,'low_dim_{}_embedding.png'.format(dim_reduction_method)),bbox_inches='tight')

	def prepare_batch(self, batch):

		batch_images = batch['data']
		batch_labels = batch['labels']
		
		# Determining batch size and image size
		batch_size = batch_images.shape[0]
		image_size = int(np.sqrt(batch_images.shape[1]/3))

		# Allocating storage for stacked image descriptors
		batch_features = np.zeros((batch_size,self.transfer_model.output_shape[1]))

		# Iterating over rows to transform them into images and computing descriptors
		for i in range(batch_size):
		
			# Rearranging data into an RGB image
			image = batch_images[i,:].reshape(3,image_size,image_size).transpose([1,2,0])
				
			# Upsampling image to original size used while training the net
			resized_image = resize(image,self.original_image_size,preserve_range=True)
			
			# Reshaping the image data to fit the net
			net_input = np.expand_dims(resized_image,axis=0)
			
			# Preprocessing that replicates the one applied at the training stage
			# (mean RGB value extraction only)
			net_input = vgg_preprocess_input(net_input)
			
			# Extracting feature vector from the last network layer
			features = self.transfer_model.predict(net_input)[0]
			
			batch_features[i,:] = features
				
			if self.debug_level > 0:
				if i % 100 == 0:
					print('Processed {} of {} images'.format(i,batch_size))

		return batch_features, batch_labels

	def prepare_data(self, input_data_path):

		print('Preparing data')
		
		all_training_features, all_test_features = None, None
		all_training_labels, all_test_labels = None, None
		
		# If features have already been extracted before, loading them
		if input_data_path.endswith('dataset_{}.npy'.format(self.model_architecture)):
			with open(input_data_path,'rb') as f:
				all_training_features = np.load(f)
				all_training_labels = np.load(f)
				all_test_features = np.load(f)
				all_test_labels = np.load(f)

		# Otherwise running the extraction process
		else:
			for path in os.listdir(input_data_path):
			
				item_path = os.path.join(input_data_path,path)
				
				if os.path.isfile(item_path) and (path.startswith('sampled_data_batch') or path.startswith('sampled_test_batch')):
					batch = unpickle_data(item_path)
				
					# Extracting features from this batch
					features, labels = self.prepare_batch(batch)
					
					if path.startswith('sampled_data_batch'):
						if all_training_features is None:
							all_training_features = features
							all_training_labels = labels
						else:
							all_training_features = np.append(all_training_features,features,axis=0)
							all_training_labels = np.append(all_training_labels,labels)
					elif path.startswith('sampled_test_batch'):
						if all_test_features is None:
							all_test_features = features
							all_test_labels = labels
						else:
							all_test_features = np.append(all_test_features,features,axis=0)
							all_test_labels = np.append(all_test_labels,labels)

					if self.debug_level > 0:
						print('Prepared data from batch {}'.format(path))

			# Saving the extracted features on disk
			with open(os.path.join(input_data_path,'dataset_{}.npy'.format(self.model_architecture)),'wb') as f:
				np.save(f,all_training_features)
				np.save(f,all_training_labels)
				np.save(f,all_test_features)
				np.save(f,all_test_labels)

		return all_training_features, all_training_labels, all_test_features, all_test_labels

	def train(self, training_data, training_labels, output_data_path):
	
		# Building model
		print('Building model')

		# Dimensionality reduction
		dim_reduction_model = PCA(whiten=True)
		dim_reduction_model.fit(training_data)
		explained_variance_ratios = dim_reduction_model.explained_variance_ratio_.cumsum()
		num_components = 200#np.argmax(explained_variance_ratios >= self.min_explained_variance_ratio)
		dim_reduction_model = PCA(n_components=num_components,whiten=True)
		reduced_training_data = dim_reduction_model.fit_transform(training_data)
	
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
					C=[0.5,1.0,1.5],				# best 1.0
					gamma=[0.001,0.0015,0.002]),	# best 0.002
				scoring=self.scoring_function,
				cv=self.num_cv_folds,
				n_jobs=self.num_parallel_jobs,
				verbose=self.debug_level)
		else:
			print('Unsupported model type. Quitting')
			
			exit(2)

		grid.fit(reduced_training_data,training_labels)

		print('Best accuracy of {} found for parameters: {}'.format(grid.best_score_,grid.best_params_))

		# Obtaining the final model
		best_model = grid.best_estimator_
		
		# Serializing final model
		print('Serializing model')
		
		model_file_name = os.path.join(output_data_path,'{}_svm_model.pkl'.format(self.model_architecture))
		
		with open(model_file_name,'wb') as f:
			pickle.dump(best_model,f)
		
		return dim_reduction_model, best_model

	def evaluate(self, dim_reduction_model, model, test_data, test_labels, metadata_path, output_data_path):
	
		# Dimensionality reduction
		reduced_test_data = dim_reduction_model.transform(test_data)

		# Reading class names
		metadata = unpickle_data(metadata_path)
		unique_class_names = [l.decode('ascii') for l in metadata[b'label_names']]
		num_classes = len(unique_class_names)
		
		# Predicting test labels
		predicted_labels = model.predict(reduced_test_data)
		
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
			'{}_svm_confusion_matrix'.format(self.model_architecture)
		)
