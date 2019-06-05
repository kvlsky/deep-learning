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

from sklearn.metrics import log_loss

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def TwoLayerCNN(image, filt1, filt2, bias1, bias2, theta3, bias3, epochs=300):
    feature_map = CNN.conv(image, filt1, bias1)
    feature_map_relu = CNN.relu(feature_map)
    max_pool = CNN.max_pooling(feature_map_relu, 2, 2)
    feature_map2 = CNN.conv(max_pool, filt2, bias2)
    feature_map_relu2 = CNN.relu(feature_map2)
    max_pool2 = CNN.max_pooling(feature_map_relu2, 2, 2) 
    # TODO: flattening ##################
    (c, h, w) = max_pool2.shape
    elements = h*w

    inputForDenseNetwork = np.zeros(c, elements, 1)
    for row in range(h):
        for col in range(w):
            inputForDenseNetwork[row+col]=max_pool2[h][w]
    DenseNetwork = CNN.FFNN(theta3)

    pred_output = DenseNetwork.forward(inputForDenseNetwork)
    y = np.random.randint(10, size=pred_output.shape)
    cel = cross_entropy(y, pred_output)

#    for i in range(epochs+1):
#        print('\nepoch {num}'.format(num = i))
#        print('\npredicted output:\n' + str(pred_output))
#        print('\nloss:\n' + str(cel))
#        print('\n---------------------------------------------')
#        DenseNetwork.train(max_pool2, y)
    (o_c, o_h, o_w) = pred_output.shape
    label = np.random.rand(o_c, o_h, o_w)
    backprop(image, label, filt1, filt2, bias1, bias2, theta3, bias3, DenseNetwork, pred_output, feature_map_relu2)

    return pred_output, cel

def backprop (image, label, filt1, filt2, bias1, bias2, theta3, bias3, DenseNetwork, pred_output, feature_map_relu2):
    # Backpropagation for the DenseLayer
    dtheta3 = DenseNetwork.backward(image, label, pred_output)
    # TODO: deflattening
    gradientMaxpool2 = maxpoolBackward(dtheta3, feature_map_relu2, 2, 2)
    print('Gradient Maxpool 2: ', str(gradientMaxpool2))
    
    

def maxpoolBackward(previousGradients, input, filter_size, stride):
    m, n_H_prev, n_W_prev, n_C_prev = input.shape
    m, n_H, n_W, n_C = previousGradients.shape
    newGradients = np.zeros(input.shape)
    for h in range(n_H):
        for w in range(n_W):
            for c in range (n_C):
                # Find the corners of the current "slice"
                vert_start = h
                vert_end = vert_start + filter_size
                horiz_start = w
                horiz_end = horiz_start + filter_size
                # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                input_slice = input[vert_start:vert_end, horiz_start:horiz_end, c]
                # Create the mask from a_prev_slice (≈1 line)
                mask = create_mask_from_window(input_slice)
                # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                newGradients[vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, previousGradients[h, w, c])
    return newGradients

def create_mask_from_window (x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = x == np.max(x)
    
    return mask
        
    
    

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

output = TwoLayerCNN(X, l1_filter, l2_filter, bias1, bias2, theta3, bias3)
print('\n==========================\nFeed-Forward CNN\n\nCross-Entropy Loss = {loss}\n\nPredicted output:\n{out}\n==========================\n'.format(loss = output[1], out = output[0]))