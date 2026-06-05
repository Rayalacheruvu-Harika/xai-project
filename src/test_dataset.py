from dataset import DamageDataset

with open("splits/train.txt") as f:
    files = [
        line.strip()
        for line in f.readlines()
    ]

dataset = DamageDataset(
    files,
    "exp_alu_steel_1_ellip_dam/train_data",
    "exp_alu_steel_1_ellip_dam/train_masks"
)

signal, mask = dataset[0]

print("Signal Shape:", signal.shape)
print("Mask Shape:", mask.shape)

print("Signal Type:", signal.dtype)
print("Mask Type:", mask.dtype)

print("Signal Min:", signal.min())
print("Signal Max:", signal.max())

print("Mask Unique:", mask.unique())