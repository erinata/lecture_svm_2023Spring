import pandas
import numpy

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

x1 = data[:,0].reshape(-1,1)
x2 = data[:,1].reshape(-1,1)
x3 = data[:,2].reshape(-1,1)


fig = pyplot.figure()
subfig = fig.add_subplot(111, projection="3d")
subfig.scatter(x1,x2,x3, c=target, depthshade = True)
pyplot.savefig("scatter_dataset_2_3d.png")



machine = svm.SVC(kernel="linear")
return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
print(return_values)


machine = svm.SVC(kernel="linear")
machine.fit(data, target)
coef = machine.coef_
intercept = machine.intercept_
print(coef)
print(intercept)


x1, x2 = numpy.meshgrid(x1,x2)
plane = -(coef[0][0] *x1 + coef[0][1] *x2 + intercept)/coef[0][2]
fig.gca().plot_surface(x1,x2, plane, alpha=0.01)

pyplot.savefig("scatter_dataset_2_3d_plane.png")











