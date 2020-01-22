import matplotlib.pyplot as plt

# Test set accuracy
left = [1, 6, 11, 16]
height = [10, 24, 36, 40]
tick_label = ['Simple - Emergency', 'CNN - Emergency', 'Simple - Cat & Dog', 'CNN - Cat & Dog']
plt.bar(left, height, tick_label=tick_label,
        width=0.8, color=['red', 'green'])
plt.xlabel('Architecture - Dataset')
plt.ylabel('Test Set Accuracy')
plt.title('Test Set Accuracy')
plt.show()
plt.savefig('Results/testSetAcc.png')


# Training set accuracy
height = [20, 30, 60, 70]
plt.bar(left, height, tick_label=tick_label,
        width=0.8, color=['red', 'green'])
plt.xlabel('Architecture - Dataset')
plt.ylabel('Training Set Accuracy')
plt.title('Training Set Accuracy')
plt.show()
plt.savefig('Results/trainingSetAcc.png')
