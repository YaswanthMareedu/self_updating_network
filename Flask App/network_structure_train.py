from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
import importlib
import os
import tensorflow as tf

grap = tf.get_default_graph()

sess = tf.Session()

from keras import backend as K
K.set_session(sess)

init_op = tf.global_variables_initializer()
sess.run(init_op)

def train():

    global grap
    tf.global_variables_initializer()
    with grap.as_default():
        
        NUM_CLASSES = len(os.listdir('C:/Users/HP/Desktop/Flask App/data/test'))
        classifier = Sequential()

            # Step 1 - Convolution
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 1), activation = 'relu'))

            # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        classifier.add(Dropout(0.25))
            # Step 1 - Convolution
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

            # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.25))
            # Step 1 - Convolution
        classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))

            # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.25))
            # Step 3 - Flattening
        classifier.add(Flatten())

            # Step 4 - Full connection
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 64, activation = 'relu'))
        classifier.add(Dropout(0.5))
            #classifier.add(Dense(output_dim = 128, activation = 'relu'))


        classifier.add(Dense(output_dim = NUM_CLASSES, activation = 'softmax'))

            #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            # Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        from keras.preprocessing.image import ImageDataGenerator

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            width_shift_range=0.02,
                                            height_shift_range=0.02,
                                            horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory('C:/Users/HP/Desktop/Flask App/data/train',
                                                            target_size = (64, 64),
                                                            batch_size = 25,
                                                            class_mode = 'categorical',
                                                            color_mode='grayscale')

        test_set = test_datagen.flow_from_directory('C:/Users/HP/Desktop/Flask App/data/test',
                                                        target_size = (64, 64),
                                                        batch_size = 25,
                                                        class_mode = 'categorical',
                                                        color_mode='grayscale')
            
        classifier.fit_generator(training_set,samples_per_epoch = NUM_CLASSES*400,nb_epoch = 25,validation_data = test_set,nb_val_samples = NUM_CLASSES*100,shuffle= True)
        #classifier.save('C:/Users/HP/Desktop/Flask App/model/model-64-64')
    
train()


'''
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('sample.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
print(result)'''