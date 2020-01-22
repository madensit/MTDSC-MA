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
trainAccCnn = [99.5, 97]

x = np.arange(len(labels))
width = 0.35

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