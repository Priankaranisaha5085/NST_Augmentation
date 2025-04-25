import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained 
## Path of trained model
save_folder_path = ""
model = load_model(f"{save_folder_path}/(drop_0.0)resnet50_model11__32b.h5") 

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
#/workspace/storage/data/originaldata/divided_data/test/',#"/home/psaha/data/augmented_data/test/"

## Provide the test data in this folder for evaulating our classification model
test_generator = test_datagen.flow_from_directory('/workspace/storage/data/augmented_data/test/',
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=False)
                                                  

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Predict the classes of the test data
Y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(test_generator.classes, y_pred)
class_names = list(test_generator.class_indices.keys())

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
#plt.show()
plt.savefig('Confusion Matrix03.png', dpi=300, bbox_inches='tight')
# Classification report
report = classification_report(test_generator.classes, y_pred, target_names=class_names)
print(report)