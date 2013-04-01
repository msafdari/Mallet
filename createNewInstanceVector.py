import os
import shutil
import sys

inputFile = open('instanceVector.txt', 'r')
alphabetFile = open('alphabet150.txt', 'r')
alines = alphabetFile.readlines()
allWords = []
for line in alines:
	allWords.append(line.strip())
lines = inputFile.readlines()
outputFile = open('newInstanceVector.txt', 'w')
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

