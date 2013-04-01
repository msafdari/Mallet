import os
import shutil
import sys


def createFile(inputDir, outputDir, filename):
	inputFile = open( inputDir + filename, 'r')
	lines = inputFile.readlines()
	outputFile = open(outputDir + filename, 'w')
	for line in lines:
		outline = ''	
		freqs = line.split(',')
		counter = 0
		num_unique = 0;
		for freq in freqs:
			if (int(freq) > 0):
				outline = outline + ' ' + str(counter) + ':' + str(freq)
				num_unique += 1
			counter += 1
		outputFile.write(str(num_unique) + outline )
		outputFile.flush()
	inputFile.close()
	outputFile.close()
	
if __name__ == '__main__':
	numpca = [25, 30, 35, 40, 45]
	numclus = [30, 50, 70, 90, 110, 130, 150]
	suffix = '_matlab_format/'
	prefix_matlab = 'folds/'
	prefix_slda = 'slda/'
	for p in numpca:
		for c in numclus:
			if c < p:
				continue
			for i in range(1,13):
				createFile(prefix_matlab, prefix_slda, str(i) + '_train.txt')
				createFile(prefix_matlab, prefix_slda, str(i) + '_test.txt')
