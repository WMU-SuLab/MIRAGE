import os
project_dir_path = os.path.dirname(__file__)
work_dirs_dir_name = 'work_dirs'
work_dirs_dir = os.path.join(project_dir_path, work_dirs_dir_name)

data_dir_name = 'data'
data_dir = os.path.join(work_dirs_dir, data_dir_name)
# data_dir = os.path.join(project_dir_path, data_dir_name)
data_divide_dir_name = os.path.join(data_dir_name, 'divide')
data_divide_dir = os.path.join(work_dirs_dir, data_divide_dir_name)
# data_divide_dir = os.path.join(project_dir_path, data_divide_dir_name)

records_dir_name = 'records'
records_dir = os.path.join(work_dirs_dir, records_dir_name)
# weights_dir_name = 'weights'
# weights_dir = os.path.join(records_dir, weights_dir_name)
checkpoints_dir_name = 'checkpoints'
checkpoints_dir = os.path.join(records_dir, checkpoints_dir_name)
logs_dir_name = 'logs'
logs_dir = os.path.join(records_dir, logs_dir_name)
load_test_name='load_test'
load_test = os.path.join(records_dir,load_test_name)
code_dir_name = 'code'
code_dir = os.path.join(records_dir, code_dir_name)

outputs_dir_name = 'outputs'
outputs_dir = os.path.join(work_dirs_dir, outputs_dir_name)
results_dir_name = 'results'
results_dir = os.path.join(work_dirs_dir, results_dir_name)

root_dir_paths = [
    work_dirs_dir,
    data_dir,
    data_divide_dir,
    records_dir,
    # weights_dir,
    checkpoints_dir,
    logs_dir,
    outputs_dir,
    # results_dir,
]

# for dir_path in root_dir_paths:
#     print(dir_path)
