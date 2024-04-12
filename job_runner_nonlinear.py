## this script is written to study the effect of variance of noise on performance of algorithm
import os, sys, subprocess, random
import numpy as np


dir_name = 'hyper_search'
# graph params
no_exp = 3
no_exp_each_run = 1
n = 1000
ds = [10,20]
ks = [1,2]
graph_type = 'ER'
sem_type = 'mlp'
search_space = [(10,30,1),(20,50,2)]
SEEDS = np.random.randint(10000, size=no_exp)
SEEDS_list = list(SEEDS)
SEEDS_sublists = [SEEDS_list[i:i + no_exp_each_run] for i in range(0, len(SEEDS_list), no_exp_each_run)]

job_directory = f"/home/changdeng/Topo_search_publish"
job_file = os.path.join(job_directory, "runner.sh")
for d in ds:
    for k in ks:
        for it in range(len(SEEDS_sublists)):
            if d==10:
                size_small = 10
                size_large = 30
                no_large_search = 1
            elif d==20:
                size_small = 20
                size_large = 50
                no_large_search = 2
            seeds = SEEDS_sublists[it]
            dims = [d,30,1]
            job_name = f"{seeds}_{n}_{d}_{int(d*k)}_{graph_type}_{sem_type}"
            args1 = f"--dir_name {dir_name} --n {n} --d {d} --s0 {int(d*k)} --graph_type {graph_type} --sem_type {sem_type} --dims {dims} --size_small {size_small} --size_large {size_large} --no_large_search {no_large_search} --seeds {seeds}"
            args1 = args1.replace('[', '').replace(']', '').replace(', ', ' ')
            

            with open(job_file, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines(f"#SBATCH --job-name='{job_name}'\n")
                #fh.writelines(f"#SBATCH --output='{job_directory}/.out/%A_{job_name}.out'\n")
                fh.writelines(f"#SBATCH --output='{job_directory}/out/%A_{job_name}.out'\n")
                fh.writelines(f"#SBATCH --error='{job_directory}/out_error/%A_{job_name}.err'\n")
                fh.writelines("#SBATCH --account=pi-naragam\n")
                fh.writelines("#SBATCH --partition=caslake\n")
                fh.writelines("#SBATCH --nodes=1\n")
                fh.writelines("#SBATCH --cpus-per-task=8\n")
                fh.writelines("#SBATCH --mem-per-cpu=4G\n")
                fh.writelines("#SBATCH --time=10:00:00\n")
                fh.writelines("\nunset XDG_RUNTIME_DIR\n")
                #fh.writelines("module load java\n")
                fh.writelines("module load python\n")
                fh.writelines("\nsource activate SCORE\n")
                fh.writelines(
                    f"srun --unbuffered python /home/changdeng/Topo_search_publish/method_runner_nonlinear.py {args1}\n")

            return_code = subprocess.run(f"sbatch {job_file}", shell=True)