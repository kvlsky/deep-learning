Übung 4 links:
1.  http://cs231n.github.io/convolutional-networks/
2.  https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
3.  https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a

# Classifier:

0. install all packages `pip -r install requirements.txt`
1. download data from https://drive.google.com/drive/folders/1MYGsij12XWBKV5UpOSpzXIoJKv6MRNj0
2. unzip Images.tar using ``tar -xvzf Images.tar`` (git bash/mingw/etc.) in classifier root directory

## Tasks:
0. RUN ON GPU https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu
1. save classification model in keras format - h5 [done]
2. create classification report [-]
3. create confusion matrix [done]
4. save plots:
    - accuracy plot [done]
    - report as heatmap [-]
    - confusion matrix [done]

## Run on GPU:
1. https://www.tensorflow.org/install/gpu
2. `pip install tensorflow-gpu`
3. run `python`
4. `sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))`
  
