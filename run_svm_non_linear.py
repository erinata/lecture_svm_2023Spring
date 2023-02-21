import pandas

from matplotlib import pyplot

from sklearn import svm
import kfold_template

dataset = pandas.read_csv("dataset/dataset_svm_2.csv")

dataset['x3'] = dataset['x1']**2 +  dataset['x2']**2

print(dataset)

target = dataset.iloc[:,0].values

data = dataset.iloc[:,1:4].values

pyplot.scatter(data[:,0], data[:,1], c=target)
pyplot.savefig("scatter_dataset_2.png")
pyplot.close()


machine = svm.SVC(kernel="linear")
return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
print(return_values)



