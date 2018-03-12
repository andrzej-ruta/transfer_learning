
from datetime import datetime
from utils.diagnostics import DataVisualizer, print_long_execution_time
from utils.error_handling import CustomArgumentParser

if __name__ == "__main__":

	# Parsing arguments
	parser = CustomArgumentParser()
	parser.add_argument('-i','--input-data-path',type=str,required=True,
		dest='input_data_path',help='Path to image data file')
	parser.add_argument('-m','--metadata-path',type=str,required=True,
		dest='metadata_path',help='Path to metadata file')
	parser.add_argument('-o','--output-data-path',type=str,required=True,
		dest='output_data_path',help='Path to the folder where the image will be saved')
	parser.add_argument('-n','--num-examples-per-class',type=int,required=False,
		default=10,dest='num_examples_per_class',help='Number of examplesper class to visualize')
	args = parser.parse_args()
	
	# Running the process
	start_time = datetime.now()

	# Creating visualization
	visualizer = DataVisualizer()
	visualizer.visualize(args.input_data_path,args.metadata_path,args.output_data_path,args.num_examples_per_class)
	
	end_time = datetime.now()
	
	# Printing duration time
	print_long_execution_time(start_time,end_time)
