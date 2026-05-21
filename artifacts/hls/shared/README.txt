Shared HLS test vectors (all variants use the same input preprocessing).

images.h              — 280 float samples [N][1][28][28]
MNIST_DATASET_IMAGE.npy — (70000, 1, 28, 28) float32, digit-sorted
MNIST_DATASET_LABEL.npy — (70000,) int64
Preprocessing: ToTensor + Normalize(mean=0.5, std=0.5) → [-1, +1]
