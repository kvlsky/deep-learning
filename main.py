import numpy as np

X = np.random.rand(3,6,6)
l1_filter = np.random.rand(1, 2, 2)
bias = 1

def conv(image, f, bias):
    D,H,W = image.shape
    d,h,w = f.shape

    padding = 0
    stride = 1

    h_out = (H - h + 2 * padding) / stride + 1
    w_out = (W - w + 2 * padding) / stride + 1

    feature_map = np.zeros((D,int(h_out),int(w_out)))

    list = []
    f_maps = []
    S = W
    s = w

    for dim in range(D):
        for i in range(S-1):
            for j in range(S-1):
                window = image[dim, j:s+j, i:s+i]

                f_map = np.dot(window, f[0])
                f_map = np.dot(f_map, bias)
                f_1 = np.sum(f_map)
                feature_map[dim,i,j] = f_1

                list.append(window)
                f_maps.append(f_map)

        # print("------------------------------'\nFilter W0", len(list))
        # print('------------------------------------\nFeature Maps',len(f_maps))
        # print('------------------------------------\nFilter\n',f[0])
        # print('------------------------------------\nF MAP\n', f_map)
        # print('------------------------------------\nF1', f_1)
        # print('------------------------------------\nFeature map shape', feature_map.shape)
        # print('------------------------------------\nFeature map\n', feature_map)
    return feature_map

feature_map = conv(X, l1_filter, bias)
print('\n==========================Feature Map==========================\n', feature_map)


def relu(feature_map):
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0, feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max(
                    [feature_map[r, c, map_num], 0])
    return relu_out

feature_map_relu = relu(feature_map)

print('\n==========================Feature Map ReLU==========================\n', feature_map_relu)
