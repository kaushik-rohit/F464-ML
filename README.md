# F464-ML
Machine Learning course, BITS PILANI

# About
The project builds the a neural network character classifier to
identify hindi character present in an image. The image may contain
more than one character. This is a case of multi-label multi-class
classification.

# Neural-Net Architecture

1) The Output layer of net contains 128 neurons corresponding to each
character of hindi.
2) The input layer is a Convolutional layer taking images of dimension
64*64. The images are resized to 64*64 so as to reduce the computation
time.
3) The hidden layer consists of Convlational and Max-Pooling Layer with
activation as Relu.
4) The activation for output layer was taken as sigmoid. Softmax doesn't
work properly in case of multi-label classification and hence sigmoid was
used with a threshold of 0.5.
h(>0.5) = 1 else 0.

# Preprocessing
1) Binarization - Images are converted to grayscale and using OTSU 
converted to binary images.
2) A combination of blur and Denoise filter is used to remove any noise.
3) Erosion is used to reduce the thickness of boundary.
4) Finally the images are resized to 64*64 dimension.
5) Keras Image Generator was also used to generate more train
images, with parameters being shift, rear, rotation.

# Results
Using the model and accuracy of 84% was achieved.

# Further Work
1) Use different architecture and preprocessing tools to achieve a higher
accuracy. ( may need collecting more data)
2) See the effect of using GAN's.
3) Implement a full translation network that does segmentation and data generation.
(i.e Hindi to English translation)
