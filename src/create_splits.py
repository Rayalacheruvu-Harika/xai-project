from pathlib import Path
import random

random.seed(42)

DATA_DIR = Path(
    "exp_alu_steel_1_ellip_dam/train_data"
)

files = sorted(DATA_DIR.glob("*.npy"))
files = [f.name for f in files]

random.shuffle(files)

n = len(files)

train = files[:28]
val = files[28:34]
test = files[34:]

Path("splits").mkdir(exist_ok=True)

with open("splits/train.txt", "w") as f:
    f.write("\n".join(train))

with open("splits/val.txt", "w") as f:
    f.write("\n".join(val))

with open("splits/test.txt", "w") as f:
    f.write("\n".join(test))

print(f"Train: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")