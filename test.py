from Classifier import Classifier
import Process

d, l = Process.get_data_for_classes(['rick','morty'])

for i,label in enumerate(l):
    if label =='rick':
        l[i] = Process.one_hot(0,2)
    else:
        l[i] = Process.one_hot(1,2)


clf = Classifier()

clf.train(d,l)
