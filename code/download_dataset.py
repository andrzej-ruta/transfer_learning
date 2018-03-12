
from datetime import datetime
from data import Downloader
from utils.diagnostics import print_long_execution_time
from utils.error_handling import CustomArgumentParser

if __name__ == "__main__":

	# Parsing arguments
	parser = CustomArgumentParser()
	parser.add_argument('-i','--input-data-url',type=str,required=False,
		default='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
		dest='input_data_url',help='URL pointing to the input images')
	parser.add_argument('-pu','--proxy-url',type=str,required=False,
		dest='proxy_url',default=None,help='Proxy server url')
	parser.add_argument('-u','--proxy-user-name',type=str,required=False,
		dest='proxy_user_name',default=None,help='User name for proxy authentication')
	parser.add_argument('-p','--proxy-password',type=str,required=False,
		dest='proxy_password',default=None,help='Password for proxy authentication')
	parser.add_argument('-o','--output-data-path',type=str,required=True,
		dest='output_data_path',help='Path to output images folder')
	parser.add_argument('-f','--data-fraction',type=float,required=False,
		dest='data_fraction',help='Fraction of data to extract')
	args = parser.parse_args()
	
	# Running the process
	start_time = datetime.now()

	# Downloading data
	downloader = Downloader(args.proxy_url,args.proxy_user_name,args.proxy_password)
	output_data_path = downloader.download(args.input_data_url,args.output_data_path)
	
	# Sampling data
	downloader.sample(output_data_path,args.data_fraction)

	end_time = datetime.now()
	
	# Printing duration time
	print_long_execution_time(start_time,end_time)
