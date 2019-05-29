import CNN
import numpy as np

X = np.random.rand(3, 6, 6)

l1_filter = np.random.rand(1, 2, 2)
bias1 = 1

l2_filter = np.random.rand(1, 2, 2)
bias2 = 1

feature_map = CNN.conv(X, l1_filter, bias1)
print('\n==========================\nFeature Map 1\n==========================\n', feature_map)

feature_map_relu = CNN.relu(feature_map)
print('\n==========================\nFeature Map ReLU 1\n==========================\n', feature_map_relu)

max_pool = CNN.max_pooling(feature_map_relu, 2, 2)
print('\n==========================\nMax Pooling 1\n==========================\n', max_pool)

feature_map2 = CNN.conv(feature_map_relu, l2_filter, bias2)
print('\n==========================\nFeature Map 2\n==========================\n', feature_map2)

feature_map_relu2 = CNN.relu(feature_map2)
print('\n==========================\nFeature Map ReLU 2\n==========================\n', feature_map_relu2)

max_pool2 = CNN.max_pooling(feature_map_relu2, 2, 2)
print('\n==========================\nMax Pooling 2\n==========================\n', max_pool2)

softmax1 = CNN.softmax(max_pool2)
print('\n==========================\nSoftmax\n==========================\n', softmax1)
