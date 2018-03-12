
from datetime import datetime
from models import DeepTransferModel
from utils.diagnostics import print_long_execution_time
from utils.error_handling import CustomArgumentParser

if __name__ == "__main__":

	# Parsing arguments
	parser = CustomArgumentParser()
	parser.add_argument('-i','--input-data-path',type=str,required=True,
		dest='input_data_path',help='Path to image data file')
	parser.add_argument('-m','--metadata-path',type=str,required=True,
		dest='metadata_path',help='Path to metadata file')
	parser.add_argument('-pu','--proxy-url',type=str,required=False,
		dest='proxy_url',default=None,help='Proxy server url')
	parser.add_argument('-u','--proxy-user-name',type=str,required=False,
		dest='proxy_user_name',default=None,help='User name for proxy authentication')
	parser.add_argument('-p','--proxy-password',type=str,required=False,
		dest='proxy_password',default=None,help='Password for proxy authentication')
	parser.add_argument('-o','--output-data-path',type=str,required=True,
		dest='output_data_path',help='Path to the model output folder')
	parser.add_argument('-n','--num-examples-per-class',type=int,required=False,
		default=50,dest='num_examples_per_class',help='Number of examplesper class to visualize')
	parser.add_argument('-d','--debug-level',type=int,required=False,
		dest='debug_level',default=0,help='Debugging level')
	args = parser.parse_args()
	
	# Running the process
	start_time = datetime.now()

	# Creating visualization
	model = DeepTransferModel('vgg16',args.debug_level,args.proxy_url,args.proxy_user_name,args.proxy_password)
	
	training_data, training_labels, test_data, test_labels = model.prepare_data(args.input_data_path)
	dim_reducer, classifier = model.train(training_data,training_labels,args.output_data_path)
	model.evaluate(dim_reducer,classifier,test_data,test_labels,args.metadata_path,args.output_data_path)
	
	end_time = datetime.now()
	
	# Printing duration time
	print_long_execution_time(start_time,end_time)
