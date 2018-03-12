
from datetime import datetime
from models import ShallowHogModel
from utils.diagnostics import print_long_execution_time
from utils.error_handling import CustomArgumentParser

if __name__ == "__main__":

	# Parsing arguments
	parser = CustomArgumentParser()
	parser.add_argument('-i','--input-data-path',type=str,required=True,
		dest='input_data_path',help='Path to image data file')
	parser.add_argument('-m','--metadata-path',type=str,required=True,
		dest='metadata_path',help='Path to metadata file')
	parser.add_argument('-o','--output-data-path',type=str,required=True,
		dest='output_data_path',help='Path to the model output folder')
	parser.add_argument('-d','--debug-level',type=int,required=False,
		dest='debug_level',default=0,help='Debugging level')
	args = parser.parse_args()
	
	# Running the process
	start_time = datetime.now()

	# Creating visualization
	model = ShallowHogModel('gradient_boosting',args.debug_level)
	
	training_data, training_labels, test_data, test_labels = model.prepare_data(args.input_data_path)
	classifier = model.train(training_data,training_labels,args.output_data_path)
	model.evaluate(classifier,test_data,test_labels,args.metadata_path,args.output_data_path)
	
	end_time = datetime.now()
	
	# Printing duration time
	print_long_execution_time(start_time,end_time)
