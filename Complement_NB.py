import pandas as pd
from sklearn.naive_bayes import ComplementNB

train_df = pd.read_csv("~/MSDA/CAP5610/code/project/digit-recognizer/train.csv")
test_df = pd.read_csv("~/MSDA/CAP5610/code/project/digit-recognizer/test.csv")

Y_df = train_df.pop('label')

binned_train_df = train_df.apply(pd.cut, bins=20, axis=0, labels=\
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

binned_test_df = test_df.apply(pd.cut, bins=20, axis=0, labels=\
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
X_df = binned_train_df


# X_df = train_df

clf = ComplementNB()

predict = clf.fit(X_df, Y_df).predict(test_df)

f = open("ComplementNB_Binned.csv", "w")
f.write("ImageId,Label\n")
for i in range(0, predict.size):
    f.write("{},{}\n".format(i+1, predict[i]))

f.close()
