import pandas as pd
from sklearn.naive_bayes import CategoricalNB

train_df = pd.read_csv("~/MSDA/CAP5610/code/project/digit-recognizer/train.csv")
test_df = pd.read_csv("~/MSDA/CAP5610/code/project/digit-recognizer/test.csv")

Y_df = train_df.pop('label')

binned_train_df = train_df.apply(pd.cut, bins=20, axis=0, labels=\
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

binned_test_df = test_df.apply(pd.cut, bins=20, axis=0, labels=\
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

X_df = train_df

clf = CategoricalNB()

# errorList = [658, 1562, 5532, 5629, 7401, 9458, 9981, 14080, 17258, 24716, 25047]

binned_test_df.loc[658,:] = 0
binned_test_df.loc[1562,:] = 0
binned_test_df.loc[5532,:] = 0
binned_test_df.loc[5629,:] = 0
binned_test_df.loc[7401,:] = 0
binned_test_df.loc[9458,:] = 0
binned_test_df.loc[9981,:] = 0
binned_test_df.loc[14080,:] = 0
binned_test_df.loc[17258,:] = 0
binned_test_df.loc[24716,:] = 0
binned_test_df.loc[25047,:] = 0



predict = clf.fit(X_df, Y_df).predict(test_df.loc[2500:5000])

f = open("CNB.csv", "w")
f.write("ImageId,Label\n")
for i in range(0, predict.size):
    f.write("{},{}\n".format(i+1, predict[i]))

f.close()
