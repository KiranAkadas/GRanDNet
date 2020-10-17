import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import indoor3d_util
from os import walk

input_folder2 = os.path.join(BASE_DIR, './data') 
#print(input_folder)

f=[]
for (dirpath, dirnames, filenames) in walk(input_folder2):
    f.extend(filenames)
    print(f)

for i in f:
    fn.append("./data/"+i)

for i in fn:
    print(i)

output_folder = os.path.join(ROOT_DIR, './data_test') 
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in fn:
    print(anno_path)
    #try:
    elements = anno_path.split('/')
    out_filename = elements[-1] # Area_1_hallway_1.npy
    print(out_filename)
    indoor3d_util.collect_point_label_nw_txt(anno_path, os.path.join(output_folder, out_filename), 'txt')
    #except:
    #    print(anno_path, 'ERROR!!')
