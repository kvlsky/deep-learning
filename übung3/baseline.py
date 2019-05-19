from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import svm
# from sklearn.metrics import recall, score, confusion_matrix
from os.path import splitext


task_name = 'speaker classification'
classes = ['andrej','lisa','leon','tobias','simon']

teram_name = 'baseline'

submission_index = 1
show_confusion = True
feature_set = 'deepscpectrum'
complexities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

feat_conf = {'deep-spectrum': (4096, 1, ',', 'infer')}

num_feat = feat_conf(feature_set[0])
ind_off = feat_conf(feature_set[1])
sep = feat_conf(feature_set[2])
header = feat_conf(feature_set[3])

features_path = '../features'
label_file = 'labels.csv'

print('======================================\nRunning' + task_name)

X_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header, usecols=range(ind_off), dtype=np.float32).values
X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header, usecols=range(ind_off), dtype=np.float32).values
X_test = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv', sep=sep, header=header, usecols=range(ind_off), dtype=np.float32).values

df_labels = pd.read_csv(label_file)

train_df = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv')
y_train = []
y_train_array = train_df['name'].values
for label in y_train_array:
    y_train.append(splitext(label)[0])


X_traindevel = np.concatenate((X_train, X_devel))
y_traindevel = np.concatenate((y_train, y_devel))




scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_devel = scaler.transform(X_devel)
X_traindevel = scaler.fit_transform(X_traindevel)
X_test = scaler.transform(X_test)

uar_scores = []
for comp in complexities:
    print("Complexity", comp)
    clf = svm.LinearSVC(c=comp, random_state=0)
    clf.fir(X_train, y_train)
    y_pred = clf.predict(X_devel)
    uar_scores.append(recall_score(y_devel, y_pred, label=classes, average='macro'))
    print('UAR on Devel', uar_scores[-1]*100)
    
    if show_confusion:
        print('Confusion matrix')
        print(classes)
        print(confusion_matrix(y_devel, y_pred, labels=classes))

optimum_complexity = complexities[np.array(uar_scores)]
print(optimum_complexity, np.max(uar_scores)*100)

clf - svm.LinearSVC(c=optimum_complexity, random_state=0)
clf.fit(X_traindevel, y_traindevel)
y_pred = clf.predict(X_test)

# erweiterte version schicken
# daten absemplen
# conf_matrix vor und nach absempling
