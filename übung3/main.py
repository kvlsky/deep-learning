from sklearn import metrics, svm, preprocessing, model_selection
import pandas as pd 
import pandas_ml as pdml

print("Running...")

data_a = pd.read_csv('data\\andrej.csv')
data_l = pd.read_csv('data\\leon.csv')
data_li = pd.read_csv('data\\lisa.csv')
data_s = pd.read_csv('data\\simon.csv')
data_t = pd.read_csv('data\\tobias.csv')

# print(data_a.shape, data_l.shape,
# data_li.shape,
# data_s.shape,
# data_t.shape)

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

X = pd.concat([data_a, data_l, data_li, data_s, data_t], ignore_index=True)
# Y = ['andrej.wav','lisa.wav','leon.wav','tobias.wav','simon.wav']
data = X.iloc[:, 1:]
labels = X.iloc[:, 0]

print(data.shape)
print(labels.shape)

train_x, test_x, train_y, test_y = model_selection.train_test_split(data, labels, test_size=0.3)
# test_x, devel_x, test_y, devel_y = model_selection.train_test_split(data, labels, test_size=3/5)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# def train_model(classifier, feature_vector_train, label, feature_vector_test):
#     # fit the training dataset on the classifier
#     classifier.fit(feature_vector_train, label)

#     # predict the labels on testation dataset
#     predictions = classifier.predict(feature_vector_test)
#     df_confusion = pd.crosstab(test_y, predictions)
    
#     return metrics.recall_score(predictions, test_y, average=None)


# recall = train_model(svm.SVC(gamma='scale'), train_x, train_y, test_x)
# print("SVM, without Over-Sampling: ", recall)


# df_confusion = pd.crosstab(test_y, y_pred)
c = [1e-5,1e-4,1e-3,1e-2,1e-1]

clf = svm.SVC(C=c[0], gamma='scale')
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)

df_confusion = pd.crosstab(test_y, y_pred)
recall = metrics.recall_score(y_pred, test_y, average='macro')

print("==============================================\nSVM, without Over-Sampling: ", recall)
print("\n==============================================\nConfusion Matrix, without Over-Sampling: ", df_confusion)










# X_np = X.astype(str).to_numpy()
# Y_np = Y.astype(str).to_numpy()
# data_l_arr = data_l.astype(str).to_numpy()

# resample_a, resample_l = ros.fit_resample(data_a_arr, data_l_arr)
# print(resample_a)
# print(resample_l) 