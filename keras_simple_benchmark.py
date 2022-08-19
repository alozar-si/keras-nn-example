#Use mlenv-tf
from time import time
import os
use_amd_gpu = 0
if(use_amd_gpu):
    print("USING AMD GPU")
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    import keras
    from keras import Sequential
    from keras.layers import Dense, Flatten
    from keras import initializers
else:
    print("USING CPU")
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras import initializers

# Create model
def create_model():
    model = Sequential() # naredimo model/ NN
    model.add(Flatten(input_shape=(28,28)))
    # prva skrita plast: 400 nevronov, 400 izhodnih nevronov, normalna porazdelitev uteži
    model.add(Dense(units=28*28, 
                    activation='relu', 
                    input_shape=(28*28,), 
                    kernel_initializer=initializers.RandomNormal(stddev=0.01)))

    #dodamo še ostale skrite plasti
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='relu'))

    #Zadnja plast je izhodna . ker iščemo skalar, vsebuje samo en nevron
    model.add(Dense(10, activation='softmax'))
    return model

# Import dataset
fashion_data = keras.datasets.mnist
(x_train, y_train) , (x_test, y_test) = fashion_data.load_data()

#Normalise dataset
x_train = x_train / 255
x_test = x_test / 255

print("----TEST 1----")
mymodel = create_model()
mymodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training 10 epochs with batch size 16384")
start_time = time()
history = mymodel.fit(x_train, y_train, epochs=10, batch_size=1024, validation_split=0.2, shuffle=True, verbose=1)
print("It took", time()-start_time, "s for 10 epochs of 1024")
start_time = time()
mymodel.predict(x_test)
print("It took", time()-start_time, "s to inference")

print("----TEST 2----")
mymodel = create_model()
mymodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training 10 epochs with full dataset")
start_time = time()
history = mymodel.fit(x_train, y_train, epochs=10, batch_size=30000, validation_split=0.2, shuffle=True, verbose=1)
print("It took", time()-start_time, "s for 10 epochs of 30000")

start_time = time()
mymodel.predict(x_test)
print("It took", time()-start_time, "s to inference")