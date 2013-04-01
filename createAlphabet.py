import string
allLetters = string.lowercase
outfile = open('alphabet.txt', 'w')
for m1 in allLetters:
	for m2 in allLetters:
		outfile.write(m1+m2+'\n')
outfile.close()
