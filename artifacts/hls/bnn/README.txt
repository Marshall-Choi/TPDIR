Variant: bnn
Checkpoint: BINARY_MNISTCNN-2026-05-21-00-10-40.pth

  conv1.weight         (16, 1, 3, 3)  int8 binary
  bn1.weight           (16,)  float32 [BN]
  bn1.bias             (16,)  float32 [BN]
  bn1.running_mean     (16,)  float32 [BN]
  bn1.running_var      (16,)  float32 [BN]
  bn1.num_batches_tracked ()  float32 [BN]
  conv2.weight         (32, 16, 3, 3)  int8 binary
  bn2.weight           (32,)  float32 [BN]
  bn2.bias             (32,)  float32 [BN]
  bn2.running_mean     (32,)  float32 [BN]
  bn2.running_var      (32,)  float32 [BN]
  bn2.num_batches_tracked ()  float32 [BN]
  conv3.weight         (32, 32, 3, 3)  int8 binary
  bn3.weight           (32,)  float32 [BN]
  bn3.bias             (32,)  float32 [BN]
  bn3.running_mean     (32,)  float32 [BN]
  bn3.running_var      (32,)  float32 [BN]
  bn3.num_batches_tracked ()  float32 [BN]
  fc.weight            (10, 3872)  float32 [FC]
  fc.bias              (10,)  float32 [FC]

weights.h — include in HLS project
npy/ — co-simulation
