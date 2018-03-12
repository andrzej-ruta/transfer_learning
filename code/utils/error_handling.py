import sys
from argparse import ArgumentParser

class CustomArgumentParser(ArgumentParser):

	def error(self, message):
		sys.stderr.write('Error: {}\n'.format(message))
		self.print_help()
		sys.exit(1)
