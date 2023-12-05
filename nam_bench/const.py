# Desc: Constants for nam_bench
# NOTE: If you add new dataset, please add it to DATASET2DEFAULT_EVALS.

DATASET2DEFAULT_EVALS = {
    "MovingDigits": ("mse", "mae", "psnr",),
    "DigitsInpainting": ("mse", "mae", "psnr",),
    "MNIST": ("accuracy", "recall", "precision", "f1", "confusion_matrix"),
}