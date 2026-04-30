import subprocess
import itertools

datasets = ['mnist', 'fashion_mnist']
schedulers = ['no_warmup', 'linear_warmup', 'cosine_warmup']
seeds = range(10)

total = 0
for dataset, scheduler, seed in itertools.product(datasets, schedulers, seeds):
    cmd = f"python train.py --dataset {dataset} --scheduler {scheduler} --seed {seed}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
    total += 1
print(f"\nDone. Ran {total} experiments.")