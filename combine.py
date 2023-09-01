import get_data
import make_model
import tensorflow as tf
from tensorflow import keras
from keras.losses import CategoricalCrossentropy
from matplotlib import pyplot as plt 

#model = Make_model2.unet(width=64, height=64, depth=None)
model = make_model.get_model(width=512, height=512, depth=64)
model.summary()

train_dataset, validation_dataset = get_data.get_data()
initial_learning_rate = 0.00001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=50, decay_rate=0.96, staircase=True
)
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=keras.optimizers.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name='Nadam',
    ),
    metrics=["acc"],
)


# Define callbacks.
#https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath = "best_model_300.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)

# Train the model, doing validation at the end of each epoch

# print(type(input_data_list), type(label_data))
#fit
epochs = 10

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    batch_size= 2,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb],
)

model.save('model_epoch_300.h5')

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

plt.savefig('epoch_300.png')