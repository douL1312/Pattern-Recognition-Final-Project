import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Input
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Φόρτωση δεδομένων
data = tf.keras.utils.image_dataset_from_directory(
  directory = 'C:\Python_repo\Final_Project\Part1\Mask_DB',
  image_size = (32, 32),
  seed = 777,
  batch_size = 29, # αριθμός που διαιρείται ακριβώς με το σύνολο των εικόνων (2088)
  shuffle = True,) 

# Κανονικοποίηση δεδομένων
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator()

# ---------------------------------------------- Ερώτημα a ----------------------------------------------
# Διαχωρισμός δεδομένων σε train_set, validation_set και test_set
train_size = round(len(data) * 0.6)
val_size = round(len(data) * 0.2)
test_size = round(len(data) * 0.2)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# ---------------------------------------------- Ερώτημα b ----------------------------------------------
# Υλοποίηση του μοντέλου
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()

    self.conv5x5 = Conv2D(filters=16, kernel_size=5, strides=1, padding='same') # συνελικτικό επίπεδο 3x3 με βήμα 1 και padding για διατήρηση διαστάσεων
    self.conv3x3_a = Conv2D(filters=32, kernel_size=3, strides=1, padding='same') # συνελικτικό επίπεδο 3x3
    self.conv3x3_b = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
    self.conv3x3_c = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
    self.batchNorm = BatchNormalization() # κανονικοποίηση των τιμών του batch, ώστε μέσος όρος=0 κι απόκλιση=1
    self.maxPool = MaxPooling2D() # μείωση διαστάσεων των χαρακτηριστικών, διατηρώντας τα πιο σημαντικά 
    self.dropout = Dropout(0.1) # επίπεδο απόρριψης με ποσοστό 0.1 για την αποφυγή του overfitting
    self.flatten = Flatten() # μετατροπή πολυδιάστατων δομών σε ένα επίπεδο διανυσμάτων
    self.fc1 = Dense(128, activation='relu') # πλήρως συνδεδεμένο επίπεδο με 128 νευρώνες και συνάρτηση ενεργοποίησης ReLU.
    self.out = Dense(1, activation='sigmoid') # πλήρως συνδεδεμένο επίπεδο εξόδου με έναν νευρώνα και συνάρτηση ενεργοποίησης Sigmoid (κατάλληλη για δυαδική ταξινόμιση)

  def call(self, inputs):
    x = self.conv5x5(inputs)
    x = self.batchNorm(x)
    x = self.maxPool(x)
    x = self.conv3x3_a(x)
    x = self.maxPool(x)
    x = self.conv3x3_b(x)
    x = self.maxPool(x)
    x = self.conv3x3_c(x)
    x = self.maxPool(x)

    x = self.flatten(x)
    x = self.fc1(x)
    x = self.dropout(x)
    x = self.out(x)
    return x
  
  def summary(self):
    x = Input(shape=(32, 32, 3))
    model = Model(inputs=[x], outputs=self.call(x))
    return model.summary()

# Σύνοψη αρχιτεκτονικής του μοντέλου
if __name__ == '__main__':
  sub = MyModel()
  sub.summary()

# Compile κι εκπαίδευση του μοντέλου και καταγραφή ιστορικού εκπαίδευσης
model = MyModel()
model.compile(optimizer='Adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
history = model.fit(train, epochs=10, validation_data=val)

# ---------------------------------------------- Ερώτημα c ----------------------------------------------
# Απεικόνιση training loss και validation loss
fig = plt.figure()
plt.plot(history.history['loss'], color='green', label='Loss')
plt.plot(history.history['val_loss'], color='orange', label='Validation Loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()

# Υπολογισμός του σφάλματος ταξινόμησης και της ακρίβειας στο test set
test_loss, test_accuracy = model.evaluate(test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Συλλογή προβλέψεων και πραγματικών ετικετών
y_true = []
y_scores = []
for batch in test.as_numpy_iterator():
  X, y = batch
  y_true.extend(y)
  y_scores.extend(model.predict(X).ravel())
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Υπολογισμός κι απεικόνιση καμπυλών precision-recall
precision, recall, _ = precision_recall_curve(y_true, y_scores)
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()

# Υπολογισμός AUC
auc_score = auc(recall, precision)
print(f'Precision-Recall AUC: {auc_score}')

# ---------------------------------------------- Ερώτημα d ----------------------------------------------
inc_use_data = tf.keras.utils.image_dataset_from_directory(
  directory = 'C:\Python_repo\Final_Project\Part1\mask_incorrect_use',
  labels = None,
  image_size = (32, 32),
  batch_size = 29,
  shuffle = False,)
inc_use_data = inc_use_data.map(lambda x: (x/255))
inc_use_data.as_numpy_iterator()
inc_use = inc_use_data.take(len(inc_use_data))

# Πρόβλεψη
preds_inc_use = []
for batch in inc_use.as_numpy_iterator():
  yhat = model.predict(batch)
  for item in yhat:
    if item >= 0.0001:
      item = 1
    if item < 0.0001:
      item = 0
    preds_inc_use.append(item)

print('-------- Predicted INCORRECT USE Mask --------')
print('People with mask: ', preds_inc_use.count(0))
print('People without mask: ', preds_inc_use.count(1))
print('Accuracy Incorrect Use: ', preds_inc_use.count(1) / (preds_inc_use.count(0) + preds_inc_use.count(1)))

zeros_list = [1] * len(preds_inc_use)
confmatr = confusion_matrix(zeros_list, preds_inc_use)
disp = ConfusionMatrixDisplay(confusion_matrix=confmatr)
disp.plot()
plt.show()