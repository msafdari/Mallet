import os
import shutil
import sys

prefix_matlab = 'folds_matlab_format/'
prefix_mallet = 'folds/'
for i in range(1,13):
	alphabetFile = open('alphabet150.txt', 'r')
	alines = alphabetFile.readlines()
	allWords = []
	for line in alines:
		allWords.append(line.strip())
		
	trainInputFile = open(prefix_matlab + str(i) + '_train.txt', 'r')
	trainLines = trainInputFile.readlines()
	trainOutputFile = open(prefix_mallet + str(i) + '_train.txt', 'w')
	for line in trainLines:
		outline = ''	
		freqs = line.split(',')
		counter = 0
		for freq in freqs:
			feature = allWords[counter] + " "
			outline = outline + feature*int(freq)
			counter = counter + 1
		outline = outline + '\n'
		trainOutputFile.write(outline)
	trainInputFile.close()
	trainOutputFile.close()
	
	testInputFile = open(prefix_matlab + str(i) + '_test.txt', 'r')
	testLines = testInputFile.readlines()
	testOutputFile = open(prefix_mallet + str(i) + '_test.txt', 'w')
	for line in testLines:
		outline = ''	
		freqs = line.split(',')
		counter = 0
		for freq in freqs:
			feature = allWords[counter] + " "
			outline = outline + feature*int(freq)
			counter = counter + 1
		outline = outline + '\n'
		testOutputFile.write(outline)	
	testInputFile.close()
	testOutputFile.close()
