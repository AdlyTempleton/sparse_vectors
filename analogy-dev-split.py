import random

random.seed(42)
with open('questions-words-dev.txt', 'w') as devfile:
    with open('questions-words-test.txt', 'w') as testfile:
        with open('questions-words.txt', 'r') as infile:
            for line in infile:
                # Section headers go in both files
                if line[0] == ':':
                    devfile.write(line)
                    testfile.write(line)
                else:
                    (devfile if random.random() < .5 else testfile).write(line)
