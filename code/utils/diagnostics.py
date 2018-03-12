import sys
import os
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from data import unpickle_data

class CustomArgumentParser(ArgumentParser):

	def error(self, message):
		sys.stderr.write('Error: {}\n'.format(message))
		self.print_help()
		sys.exit(1)

class DataVisualizer():

	def __init__(self):
		pass

	def visualize(self, input_data_path, metadata_path, output_data_path, num_examples_per_class):
		
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
		
		# Building random sample
		image_number = 1
		
		for class_id in range(num_classes):
			class_indices = labels == class_id
			class_images = images[class_indices,:]
			
			total_num_images = class_images.shape[0]
			selected_indices = np.random.choice(total_num_images,num_examples_per_class,replace=False)
			sample_class_images = class_images[selected_indices,:]
			
			for i in range(num_examples_per_class):
				plt.subplot(num_classes,num_examples_per_class,image_number)
				plt.imshow(sample_class_images[i,:].reshape(3,image_size,image_size).transpose([1,2,0]))
				plt.axis('off')
				image_number += 1
			
			plt.text(-570,class_id+image_size/2,unique_class_names[class_id],fontsize=12)

		plt.yticks(np.arange(num_classes),unique_class_names)
		plt.savefig(os.path.join(output_data_path,'sample_images.png'),bbox_inches='tight')
	
def print_long_execution_time(start_time, end_time, logger=None):
    duration = end_time-start_time
    seconds_elapsed = duration.total_seconds()
    duration_sec = int(seconds_elapsed)
    duration_hrs = duration_sec // 3600
    duration_sec = duration_sec % 3600
    duration_min = duration_sec // 60
    duration_sec = duration_sec % 60

    message = '\nDone. Processing took {}h {}m {}s'.format(duration_hrs, duration_min, duration_sec)
    
    if logger:
        logger.info(message)
    else:
        print(message)

def print_short_execution_time(start_time, end_time, logger=None):
    duration = end_time-start_time
    seconds_elapsed = duration.total_seconds()
    
    message = '\nDone. Processing took {0:.2f}ms'.format(seconds_elapsed*1000)

    if logger:
        logger.info(message)
    else:
        print(message)
