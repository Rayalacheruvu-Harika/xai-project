import pandas as pd


metrics = pd.read_csv(
    "results/test_metrics.csv"
)

xai = pd.read_csv(
    "results/xai_metrics.csv"
)

noise = pd.read_csv(
    "results/noise_xai.csv"
)

print("\nMODEL PERFORMANCE")
print("-----------------")

print(
    "Dice:",
    metrics["dice"].mean()
)

print(
    "IoU:",
    metrics["iou"].mean()
)

print("\nXAI PERFORMANCE")
print("-----------------")

print(
    "GradCAM Alignment:",
    xai["gradcam_alignment"].mean()
)

print(
    "IG Alignment:",
    xai["ig_alignment"].mean()
)

print("\nNOISE ROBUSTNESS")
print("-----------------")

print(noise)