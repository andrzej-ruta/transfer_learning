import sys
import os
from urllib.request import ProxyHandler, build_opener, install_opener, urlretrieve
from zipfile import ZipFile
import tarfile
import pickle
import numpy as np

class Downloader():

	def __init__(self, proxy_url=None, proxy_user_name=None, proxy_password=None):
	
		if proxy_url:

			# Setting up proxy
			proxy = ProxyHandler({'https': '{}:{}@{}'.format(proxy_user_name,proxy_password,proxy_url)})
			opener = build_opener(proxy)
			install_opener(opener)

	def show_download_progress(self, count, block_size, total_size):

		# Determining percentage completion
		completion_percentage = float(count*block_size)/total_size

		message = '\r- Download progress: {0:.1%}'.format(completion_percentage)

		# Printing progress message
		sys.stdout.write(message)
		sys.stdout.flush()

	def download(self, input_data_url, output_data_path):

		file_name = input_data_url.split('/')[-1]
		output_file_path = os.path.join(output_data_path,file_name)

		if not os.path.exists(output_file_path):
		
			# Creating output directory if it does not exist
			if not os.path.exists(output_data_path):
				os.makedirs(output_data_path)

			# Downloading the data file
			file_path, _ = urlretrieve(url=input_data_url,
				filename=output_file_path,
				reporthook=self.show_download_progress)
		
			print('\nDownload completed')
		else:
			print('File already downloaded')
		
		extracted_archive_path = None
		
		if output_file_path.endswith('.zip'):
		
			# Extracting zip archive
			archive = ZipFile(file=output_file_path,mode='r')
			archive.extractall(output_data_path)
			extracted_archive_path = os.path.join(output_data_path,archive.infolist()[0].filename)
		
			print('Archive unpacked')
		
		elif output_file_path.endswith(('.tar.gz','.tgz')):
		
			# Extracting tar-ball
			archive = tarfile.open(name=output_file_path,mode='r:gz')
			archive.extractall(output_data_path)
			extracted_archive_path = os.path.join(output_data_path,archive.getmembers()[0].name)

			print('Archive unpacked')
		else:
			print('Archive file format not supported')
		
		return extracted_archive_path

	def sample(self, input_data_path, data_fraction=1.0):
	
		for path in os.listdir(input_data_path):
		
			item_path = os.path.join(input_data_path,path)
			
			if os.path.isfile(item_path) and (path.startswith('data_batch') or path.startswith('test_batch')):
				batch = unpickle_data(item_path)

				# Building random sample
				batch_size = batch[b'data'].shape[0]

				if path.startswith('data_batch'):
					sample_size = int(data_fraction*batch_size)
				else:
					sample_size = batch_size

				selected_indices = np.random.choice(batch_size,sample_size,replace=False)
				sample_data = batch[b'data'][selected_indices,:]
				sample_labels = list(np.array(batch[b'labels'])[selected_indices])
				sample_batch = {'batch_label': batch[b'batch_label'], 'data': sample_data, 'labels': sample_labels}
				
				# Serializing back the sampled data
				if path.startswith('data_batch'):
					sample_batch_path = os.path.join(input_data_path,path.replace('data_batch','sampled_data_batch'))
				else:
					sample_batch_path = os.path.join(input_data_path,path.replace('test_batch','sampled_test_batch'))
				
				pickle_data(sample_batch,sample_batch_path)

		print('Sampling completed')
				
def unpickle_data(path):

	with open(path,'rb') as f:
		return pickle.load(f,encoding='bytes')

def pickle_data(data, path):

	with open(path,'wb') as f:
		pickle.dump(data,f)
		