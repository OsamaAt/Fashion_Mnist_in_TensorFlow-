import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

fmnist=tf.keras.datasets.fashion_mnist
(training_images , train_label) , (testing_images ,test_label) = fmnist.load_data()

index=5
np.set_printoptions(linewidth=320)
print(f'Training_images : {training_images[index]}')
print(f'label_images : {train_label[index]}')


plt.imshow(training_images[index])
plt.colorbar()
plt.show()

training_images = training_images / 255.0
testing_images = testing_images / 255.0

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,log=None):
        if log['loss'] < 0.4:
            print('\n loss is under 0.4 no cancelling training')
            self.model.stop_training=True

# #Add class HERE

model=tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128 , activation=tf.nn.relu),
    tf.keras.layers.Dense(10 , activation=tf.nn.softmax)
])

inputs=np.array([[1.0,3.0,4.0,2.0]])
inputs=tf.convert_to_tensor(inputs)
print(f'Inputs To SoftMax Function {inputs.numpy()}')

outputs=tf.nn.softmax(inputs)
print(f'outputs To SoftMax Function {outputs.numpy()}')

sum=tf.reduce_sum(outputs)
print(f'the sum of the outputs is {sum}')

prediction=np.argmax(outputs)
print(f'class with highest outputs : {prediction}')

model.compile(optimizer=tf.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(training_images , train_label , epochs=5 , callbacks=[myCallback()] )

model.evaluate(testing_images , test_label)
pred=model.predict(testing_images)
print(pred)