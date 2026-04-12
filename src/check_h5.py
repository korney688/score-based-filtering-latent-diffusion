import h5py

with h5py.File("e:/ITMO/НИР Хватов/2 семместр/project/experiments/exp_001/data/dataset_noisy/dataset_noisy.h5", "r") as f:
    print(list(f.keys()))
    print(f["dataset"].shape)
    print(f["dataset"].dtype)