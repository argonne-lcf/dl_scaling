import os
import shutil
from tqdm import tqdm

root_dir = "/grand/datascience/zhaozhenghao/datasets/cosmoflow/tf_v2_256"
out_dir = "/grand/datascience/zhaozhenghao/datasets/cosmoflow/tf_v2"

for data_folder in os.listdir(root_dir):
    with open(os.path.join(root_dir, data_folder, "files_data.lst"), "r") as input_file:
        file_list_path = input_file.readlines()
    
    result_names = []
    result_size = []
    for line in tqdm(sorted(file_list_path)):
        file_name, size = line.split(" ")

        result_names.append(file_name.strip())
        result_size.append(int(size.strip()))

        shutil.copy(os.path.join(root_dir, data_folder, file_name.strip()), os.path.join(out_dir, data_folder, file_name.strip()))
        shutil.copy(os.path.join(root_dir, data_folder, file_name.strip()+'.idx'), os.path.join(out_dir, data_folder, file_name.strip()+'.idx'))

        for i in range(1, 10):
            file_name_ = file_name.replace("_0","_"+str(i))
            for j in range(1, 10):
                file_name_gen = file_name_.replace("2019-03", "2019-0"+str(j))

                result_names.append(file_name_gen.strip())
                result_size.append(int(size.strip()))

                shutil.copy(os.path.join(root_dir, data_folder, file_name.strip()), os.path.join(out_dir, data_folder, file_name_gen.strip()))
                shutil.copy(os.path.join(root_dir, data_folder, file_name.strip()+'.idx'), os.path.join(out_dir, data_folder, file_name_gen.strip()+'.idx'))
    
    with open(os.path.join(out_dir, data_folder, "files_data.lst"), "w") as output_file:
        for name, size in zip(result_names, result_size):
            output_file.write(name + " " + str(size) + "\n")
