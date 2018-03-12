import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def plot_confusion_matrix(cm, labels, num_classes, chart_width, chart_height, output_path, output_file_name):

	# Normalizing confusion matrix
	norm_cm = []

	for i in cm:
		a = 0
		tmp_array = []
		a = sum(i,0)
		
		for j in i:
			tmp_array.append(float(j)/float(a))
		
		norm_cm.append(tmp_array)
        
	# Computing overall accuracy
	accuracy = np.trace(cm)/np.sum(cm)*100

	fig = plt.figure(figsize=(chart_width,chart_height))
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)

	X, Y = np.mgrid[0:num_classes+1,0:num_classes+1]
	norm = matplotlib.colors.Normalize(vmin=0.0,vmax=1.0,clip=True)
	pcm = ax.pcolormesh(X,Y,np.fliplr(np.transpose(np.array(norm_cm))),norm=norm,cmap=plt.cm.jet)

	width, height = cm.shape

    # Adding axis labels
	for x in range(width):
		for y in range(height):
			ax.annotate(str(np.round(norm_cm[x][y],2)),
				xy=(y+0.5,width-x-0.5),
				horizontalalignment='center',
				verticalalignment='center',
				fontsize=8)

	cb = fig.colorbar(pcm)
    
	plt.xticks([x+0.5 for x in range(width)],labels,fontsize=8,rotation=90)
	plt.yticks([y+0.5 for y in range(height-1,-1,-1)],labels,fontsize=8)
	plt.xlabel('Predicted categories',fontsize=12)
	plt.ylabel('True categories',fontsize=12)
	plt.title('Accuracy: {0:.2f}%'.format(accuracy),fontsize=16)

	# Saving confusion matrix asimage
	plt.savefig(os.path.join(output_path,output_file_name + '.png'),
		format='png',
		bbox_inches='tight'
	)
	plt.close()
