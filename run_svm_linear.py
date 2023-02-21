import pandas

from matplotlib import pyplot

from sklearn import svm
import kfold_template

dataset = pandas.read_csv("dataset/dataset_svm_1.csv")

print(dataset)

target = dataset.iloc[:,0].values

data = dataset.iloc[:,1:3].values

pyplot.scatter(data[:,0], data[:,1], c=target)
pyplot.savefig("scatter_dataset_1.png")
pyplot.close()

machine = svm.SVC(kernel="linear")
return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
print(return_values)



