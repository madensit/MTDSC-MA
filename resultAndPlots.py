import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


labels = ['Emergency', 'Cat and Dog']
testAccSimple = [95, 91]
testAccCnn = [96, 85]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, testAccSimple, width, label='Simple')
rects2 = ax.bar(x + width / 2, testAccCnn, width, label='CNN')

ax.set_ylabel('Accuracy')
ax.set_title('Test set Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('Results/testSetAcc.png')
plt.show()


trainAccSimple = [100, 90]
trainAccCnn = [99.5, 96.5]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, trainAccSimple, width, label='Simple')
rects2 = ax.bar(x + width / 2, trainAccCnn, width, label='CNN')

ax.set_ylabel('Accuracy')
ax.set_title('Train set Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('Results/trainingSetAcc.png')
plt.show()



trainAccSimpleNoDr = [100, 100]
trainAccCnnNoDr = [100, 100]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, trainAccSimpleNoDr, width, label='Simple without Dropout')
rects2 = ax.bar(x + width / 2, trainAccCnnNoDr, width, label='CNN  without Dropout')

ax.set_ylabel('Accuracy')
ax.set_title('Train set Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('Results/trainingSetAccNoDr.png')
plt.show()


testAccSimpleNoDr = [96, 96]
testAccCnnNoDr = [98, 90]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, testAccSimpleNoDr, width, label='Simple without Dropout')
rects2 = ax.bar(x + width / 2, testAccCnnNoDr, width, label='CNN without Dropout')

ax.set_ylabel('Accuracy')
ax.set_title('Test set Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('Results/testSetAccNoDr.png')
plt.show()


trainAccSimpleDr2 = [100, 90]
trainAccCnnDr2 = [96, 77]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, trainAccSimpleDr2, width, label='Simple with greater Dropout')
rects2 = ax.bar(x + width / 2, trainAccCnnDr2, width, label='CNN with greater Dropout')

ax.set_ylabel('Accuracy')
ax.set_title('Train set Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('Results/trainSetAccDr2.png')
plt.show()


testAccSimpleDr2 = [96, 91]
testAccCnnDr2 = [96, 75]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, testAccSimpleDr2, width, label='Simple with greater Dropout')
rects2 = ax.bar(x + width / 2, testAccCnnDr2, width, label='CNN with greater Dropout')

ax.set_ylabel('Accuracy')
ax.set_title('Test set Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('Results/testSetAccDr2.png')
plt.show()


