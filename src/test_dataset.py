from datasets import my_dataset

ds = my_dataset(
    h5_path="synthetic.h5",
    data_key="dataset",
    in_memory=False
)

print("len:", len(ds))

sample = ds[0]
print("type:", type(sample))
print("shape:", sample.shape)
print("dtype:", sample.dtype)