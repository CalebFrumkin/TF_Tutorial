#tutorial from https://www.datacamp.com/community/tutorials/tensorflow-tutorial
import tensorflow as tf
import numpy as np
import os
import skimage
import matplotlib.pyplot as plt
import random

def load_data(data_directory):
    directories=[d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory,d))]
    labels=[]
    images=[]
    for d in directories:
        label_directory=os.path.join(data_directory,d)
        file_names=[os.path.join(label_directory,f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

def plot(images):
    signs=[300,2250,3650,4000]

    for i in range(len(signs)):
        plt.subplot(1,4,i+1)
        plt.axis('off')
        plt.imshow(images[signs[i]],cmap='gray')
        plt.subplots_adjust(wspace=0.5)

    plt.show()


ROOT_PATH="F:/users/caleb/documents/1_important/hackingaround/machinelearning/tensorflow"
train_data_directory=os.path.join(ROOT_PATH, "Training")
test_data_directory=os.path.join(ROOT_PATH, "Testing")

images,labels=load_data(train_data_directory)

images=np.array(images)
images28=[skimage.transform.resize(image, (28,28)) for image in images]
images28=np.array(images28)
images28=skimage.color.rgb2gray(images28)
#create placeholders
x=tf.placeholder(dtype=tf.float32,shape=[None,28,28])
y=tf.placeholder(dtype=tf.int32, shape=[None])

images_flat=tf.contrib.layers.flatten(x)
#fully connected layer
logits=tf.contrib.layers.fully_connected(images_flat,62,tf.nn.relu)
#loss function
loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits))
#optimizer, minimizes loss function
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred=tf.argmax(logits,1)
#accuracy metric defined
accuracy=tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#training
tf.set_random_seed(1234)
sess=tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
#    print('EPOCH',i)
    _,accuracy_val=sess.run([train_op,accuracy],feed_dict={x:images28,y:labels})
    if i%10==0:
        print("Loss: ",loss)
#    print('DONE WITH EPOCH')




test_images, test_labels=load_data(test_data_directory)
#resize to 28x28 pixels
test_images28=[skimage.transform.resize(image,(28,28)) for image in test_images]

#make grayscale
test_images28=skimage.color.rgb2gray(np.array(test_images28))

#make predictions
predicted=sess.run([correct_pred], feed_dict={x: test_images28})[0]

#sum all correct matches
match_count= sum([int(y==y_) for y, y_ in zip(test_labels, predicted)])

#find percentage/accuracy
accuracy = match_count / len(test_labels)

#print percentage
print("Accuracy: {:,3f}",format(accuracy))
