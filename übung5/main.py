import CNN
import numpy as np

X = np.random.rand(3, 12, 12)

l1_filter = np.random.rand(1, 2, 2)
bias1 = 1

l2_filter = np.random.rand(1, 2, 2)
bias2 = 1

# theta3 = l1_filter
theta3 = np.random.rand(3, 2, 2)
bias3 = 1

def TwoLayerCNN(image, filt1, filt2, bias1, bias2, theta3, bias3):
    feature_map = CNN.conv(image, filt1, bias1)
    feature_map_relu = CNN.relu(feature_map)
    max_pool = CNN.max_pooling(feature_map_relu, 2, 2)
    feature_map2 = CNN.conv(max_pool, filt2, bias2)
    feature_map_relu2 = CNN.relu(feature_map2)
    max_pool2 = CNN.max_pooling(feature_map_relu2, 2, 2) 
    
    DenseNetwork = CNN.FFNN(theta3)

    pred_output = DenseNetwork.forward(max_pool2)
    y = np.random.randint(10, size=pred_output.shape)

    mse = np.mean(np.square(y - DenseNetwork.forward(max_pool2)))

    # for i in range(10):
    #     print('epoch {num}'.format(num = i))
    #     print('\npredicted output: ' + str(pred_output))
    #     print('\nloss: \n' + str(mse))
    #     print('\n---------------------------------------------')
    #     DenseNetwork.train(max_pool2, y)
    
    # prediction = DenseNetwork.predict()

    return mse, pred_output

feature_map = CNN.conv(X, l1_filter, bias1)
print('\n==========================\nFeature Map 1\n==========================\n', feature_map)

feature_map_relu = CNN.relu(feature_map)
print('\n==========================\nFeature Map ReLU 1\n==========================\n', feature_map_relu)

max_pool = CNN.max_pooling(feature_map_relu, 2, 2)
print('\n==========================\nMax Pooling 1\n==========================\n', max_pool)

feature_map2 = CNN.conv(max_pool, l2_filter, bias2)
print('\n==========================\nFeature Map 2\n==========================\n', feature_map2)

feature_map_relu2 = CNN.relu(feature_map2)
print('\n==========================\nFeature Map ReLU 2\n==========================\n', feature_map_relu2)

max_pool2 = CNN.max_pooling(feature_map_relu2, 2, 2)
print('\n==========================\nMax Pooling 2\n==========================\n', max_pool2)

softmax1 = CNN.softmax(max_pool2)
print('\n==========================\nSoftmax\n==========================\n', softmax1)

output = TwoLayerCNN(X, l1_filter, l2_filter, bias1, bias2, theta3, bias3)
print('\n==========================\nFeed-Forward CNN\n\nMSE = {mse}\nPredicted output:\n{out}\n==========================\n'.format(mse = output[0], out = output[1]))

