from sklearn import metrics, svm, preprocessing, model_selection
import pandas as pd 

print("Running...")

# read data from .csv to pandas DataFrame
data_a = pd.read_csv('data\\andrej.csv')
data_l = pd.read_csv('data\\leon.csv')
data_li = pd.read_csv('data\\lisa.csv')
data_s = pd.read_csv('data\\simon.csv')
data_t = pd.read_csv('data\\tobias.csv')

# Concatenate all DataFrames
X = pd.concat([data_a, data_l, data_li, data_s, data_t], ignore_index=True)
# input data
data = X.iloc[:, 1:]
# labels (classes)
labels = X.iloc[:, 0]

print(data.shape)
print(labels.shape)

# Data split
train_x, test_x, train_y, test_y = model_selection.train_test_split(data, labels, test_size=0.3)
# test_x, devel_x, test_y, devel_y = model_selection.train_test_split(data, labels, test_size=3/5)

# Labels binary encoder
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# complexities
c = [1e-5,1e-4,1e-3,1e-2,1e-1]

# define classifier
clf = svm.SVC(C=c[0], gamma='scale')
# fit data into classifier
clf.fit(train_x, train_y)
# make predictions
y_pred = clf.predict(test_x)

# confusion matrix
df_confusion = pd.crosstab(test_y, y_pred)
# UAR score
recall = metrics.recall_score(y_pred, test_y, average='macro')

print("==============================================\nSVM, without Over-Sampling: ", recall)
print("\n==============================================\nConfusion Matrix, without Over-Sampling: ", df_confusion)
