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
		for freq in freqs:
			feature = allWords[counter] + " "
			outline = outline + feature*int(freq)
			counter = counter + 1
		outline = outline + '\n'
		outputFile.write(outline)
	inputFile.close()
	outputFile.close()
	
if __name__ == '__main__':
	alphabetFile = open('alphabet150.txt', 'r')
	alines = alphabetFile.readlines()
	allWords = []
	for line in alines:
		allWords.append(line.strip())
	numpca = [25, 30, 35, 40, 45]
	numclus = [30, 50, 70, 90, 110, 130, 150]
	suffix = '_matlab_format/'
	prefix = 'folds/'
	for p in numpca:
		for c in numclus:
			if c < p:
				continue
			dirname = prefix + 'p' + str(p) + '_c' + str(c)
			dirname_matlab = dirname + suffix;
			dirname += '/'
			for i in range(1,13):
				createFile(dirname_matlab, dirname, str(i) + '_train.txt')		
				createFile(dirname_matlab, dirname, str(i) + '_test.txt')
