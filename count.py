import string

chars = []

with open("output.txt", "r") as f:
	for c in f.read():
		chars.append(c)

num_chars = len(chars)
num_correct = 0;
num_incorrect = 0

digit = "0123456789"
for c in chars:
	if c in digit:
		num_correct += 1
	else:
		num_incorrect += 1

print(num_correct)
print(num_chars)