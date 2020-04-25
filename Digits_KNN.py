import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv("~/MSDA/CAP5610/code/project/digit-recognizer/train.csv")
test_df = pd.read_csv("~/MSDA/CAP5610/code/project/digit-recognizer/test.csv")

print(train_df.shape)
print(test_df.shape)

# numTrainRow = train_df.shape[0]
# numTrainCol = train_df.shape[1]
# numTestRow = test_df.shape[0]
# numTestCol = test_df.shape[1]

Y_df = train_df.pop('label')
X_df = train_df


# binned_train_df = train_df.apply(pd.cut, bins=20, axis=0, labels=\
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
#
# binned_test_df = test_df.apply(pd.cut, bins=20, axis=0, labels=\
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
#
# X_df = binned_train_df




print(X_df.head())
print(Y_df.head())

classifier = KNeighborsClassifier(n_neighbors=1)
# classifier = KNeighborsClassifier(n_neighbors=3)
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier = KNeighborsClassifier(n_neighbors=7)


classifier.fit(X_df, Y_df)

predict = classifier.predict(test_df)

f = open("KNN-1Neighbor_Binned.csv", "w")
f.write("ImageId,Label\n")
for i in range(0, predict.size):
    f.write("{},{}\n".format(i+1, predict[i]))

f.close()
