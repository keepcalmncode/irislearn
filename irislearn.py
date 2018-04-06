from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
features = iris.data
labels = iris.target
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=.5)
my_classifier = KNeighborsClassifier()
my_classifier.fit(features_train , labels_train)
prediction = my_classifier.predict(features_test)
print(accuracy_score(labels_test,prediction))
#add your test data below
iris1 = [[4.0,3.0,2.0,1.0]]
#let your trained model predict the result
iris_prediction = my_classifier.predict(iris1)
#print the predicted values
if iris_prediction[0] == 0:
	print('setosha')
if iris_prediction[0] == 1:
	print('versicolor')
if iris_prediction[0] == 2:
	print('virginica')


