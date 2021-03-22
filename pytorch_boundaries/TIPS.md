# Notes on HED reproduced

Generic thoughts
Rule-0: Use pytorch. Seriously, do it.

1. VGG16 works as a better backbone than VGG16-bn
2. Adam converges well, but SGD-Momentum is what produced SOTA results
3. Too much augmentation maybe troublesome, static rotation+flip+scaling worked best for BSDS500
4. Choose low learning rate on backbone if pretrained

Specific implementational tips
1. Learning rate schedule important, 10,000 steps once lr/=10
2. Weight initialization is crucial, make sure that enough thoughts go into this
3. Extract layers from torchvision models for pretrained nets
4. No random cropping here; train models in full resolution (around 500x600 is max resolution)
5. IMPORTANT: Make sure loss for one example is not scaled up. E.g. when implementing gradient accumulation for large batches, backward() should be called at each sub-minibatch with the appropriate scaling
6. torchvision takes RGB input that is in [0, 255] range, standardized using pretrained dataset mean and stddev
7. Initial zero padding is important; each stage's output is cropped to the unpadded input resolution
8. *True class label should be set to 1*. For BSDS500, all label values <0.5 are set to 0. and all else set to 1. otherwise, if labels are real-valued, then prediction probabilities will be undercalibrated
9. Weight decay set to 2e-4