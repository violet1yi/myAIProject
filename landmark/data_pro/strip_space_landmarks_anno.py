#change the split format of origional label file, just one space between each element
import os
import re

pro_file = './anno/list_landmarks_align_celeba.txt'
out_file = open('./anno/list_landmarks__align_celeba_strip.txt', 'w')
count = 0

with open(pro_file) as input:
    for line in input:
        count += 1
        line = line.strip()
        line_without_space = re.split(r'\s+', line)
        new_line = ' '.join(line_without_space) + '\n'
        out_file.write(new_line)
        if count%500 == 0:
            print('%s lines has been processed'%(count))


out_file.close()







