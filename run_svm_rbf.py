import pandas
import numpy

from matplotlib import pyplot

from sklearn import svm
import kfold_template

dataset = pandas.read_csv("dataset/dataset_svm_2.csv")


target = dataset.iloc[:,0].values

data = dataset.iloc[:,1:3].values

machine = svm.SVC(kernel="poly")
return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
print(return_values)

