# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./assets/classes_example.png "Visualization"
[image2]: ./assets/classes_distribution.png "Distributions"
[image3]: ./assets/grayscaled.png "Grayscaling"
[image4]: ./assets/web_images.png "Web images"
[image5]: ./assets/predictions.png "Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32x32x3): 32 pixels width and height, with 3 color channels
* There are 43 unique classes in the traffic signs data set

#### 2. Include an exploratory visualization of the dataset.

Let us begin by displaying an instance of every single class in the data set. We can note that pictures have very different lighting conditions.

![alt text][image1]

Next, we can plot the distribution of the classes to see how balanced is the dataset.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because since, although traffic signs varies in color in real life, their differences in shape and patterns seem enough to learn for the neural network.
The code for this transformation can be found in `cell 9` of the notebook.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a second step, I normalized the image (`cell 10` of the notebook) data to make it easier for the network to learn. Indeed, without normalization, the learning rate of the gradient descent could compare features that have totally different scale, hence comparing "apples and oranges", and could potentially have more trouble finding an optimum.

Unfortunately, I did not generate additional data yet, the results of the testing set being sufficiently good. However, adding random noise, shifting, flipping, rotating or projection transforms are processing that I intend to do later.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted simply of the original LeNet architecture to which I added dropout and increased the output dimension of each layer. I figured that we needed more dimension to capture the information from the images. Indeed, traffic sign picture are more rich than the one from the MNIST, hence having more dimensions to discriminate against seemed logical.
In the end, the final network is the following:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscaled image   							|
| **Layer 1**   |   |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 kernel size,  outputs 14x14x32 				|
| **Layer 2**   |   |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU   |   |
| Max pooling   | 2x2 stride, valid padding, outputs 5x5x64   |
| Flatten   | inputs 5x5x64; outputs 1600   |
| **Layer 3**   |   |
| Fully connected		| outputs 800 |
| Dropout   | drops 50% of neurons   |
| RELU   |   |
| **Layer 4**   |   |
| Fully connected		| outputs 400 |
| Dropout   | drops 50% of neurons   |
| RELU   |   |
| **Layer 5**   |  Not present in the original LeNet architecture |
| Fully connected		| outputs 43 |
| Softmax				|  Outputs classes probability vectors  |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the default optimizer (ADAM) after having tested AdaDelta as well and having lower performance. I noticed that when using smaller batch size than the original implementation, the validation accuracy increased, so I went with a batch size of 64. I was able to get sufficiently quick gradient descent with a learning rate of 0.0006, compared to the 0.001 that seemed to fluctuate more. Finally, regarding the number of epochs, after a point, the validation accuracy did not increase anymore and only fluctuate around the 94% validation accuracy threshold, so I sticked with 25 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 94.3%
* test set accuracy of 93.3%


If a well known architecture was chosen:
* What architecture was chosen?

As suggested in the lesson, I chose to start with the LeNet architecture and I believe it was a good starting point. As explained aboev, I tweaked and added parameters a bit to be able to increase the performance on the traffic signs.

Tried and kept :

  - Increased the number of outputs in every layers to increase the information passed on to the model until the last output layer, so the model has more information to discriminate images
  - Added dropout


Tried and discarded :
  - Tried to add batch normalization, add trouble restore the batch distribution at test time
  - Tried to add a 6th layer (a 3rd fully connected layer) but did not really improve the performance



* Why did you believe it would be relevant to the traffic sign application?

The LeNet architecture was first designed to classify handwritten digits. There are obivous differences in the data sets: notably, the handwritten digits are only in black and white, there is no noise in the image while there is a bit noise in the traffic sign (not too much since the pictures are cropped and centered on the traffic sign itself) and the pictures are colored.
However, both datasets represent very standardized objects, with simple shapes to identify and grayscaling pictures resolves one of the two main differences.
Hence, by tweaking a bit the parameters, and for instance increasing the output dimension, we were able to capture enough information to accurately predict a traffic sign class.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The performance is higher on the training set, reaching almost 100%. It can let us to think that the model may be overfitting a bit. However, the validation and test set are consistent and are not much much lower than the training so we can be reasonably confident about our model. A way to improve the model would be to reduce this overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

All these pictures seem pretty easy to classify, the only challenge could be that the original image were not squares, so rescaling induce a deformation. Will that mislead the model ?

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road      		| Priority road   									|
| Right-of-way at the next intersection     			| Right-of-way at the next intersection  										|
| No entry					| No entry											|
| Stop sign	      		| Stop sign					 				|
| Give way			| Give way      							|


The model was able to correctly guess 5 out of the 5 new traffic signs, which gives an accuracy of 100%. This is in par with the accuracy found in the training, validation and test set. It is a bit higher, but the image are well exposed, and there is no particular difficulties on these images. I expected them to be correcly predicted. Moreover, 5 images is not a number of samples big enough to draw conclusions, the test set accuracy is undoubtedly more relevant.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in `cell 29` of the notebook.

For all images, my model is absolutely sure of the class to predict:
For the first image, the model is absolutely sure that this is a priority road sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Priority road   									|
| .00     				| Stop 										|
| .00					| No vehicles											|
| .00	      			| Ahead only					 				|
| .00				    | Road work      							|


For the second image, the model is absolutely sure that this is a right-of-way at the next intersection sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Right-of-way at the next intersection |
| .00     				| Roundabout mandatory 										|
| .00					| Beware of ice/snow											|
| .00	      			| Double curve					 				|
| .00				    | Dangerous curve to the left      							|


For the third image, the model is absolutely sure that this is a No entry sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| No entry   									|
| .00     				| Stop 										|
| .00					| End of all speed and passing limits											|
| .00	      			| Speed limit (30 km/h)					 				|
| .00				    | Speed limit (50km/h)      							|

For the fourth image, the model is absolutely sure that this is a Stop sign (probability of 0.995). The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.995         			| Stop   									|
| .003     				| Turn left ahead 										|
| .001					| Speed limit (80 km/h)											|
| .0001	      			| Speed limit (60 km/h)					 				|
| .00006				    | Keep right      							|

For the fifth and last image, the model is absolutely sure that this is a Yield sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Yield   									|
| .00     				| Right-of-way at the next intersection |
| .00					| Speed limit (30 km/h)											|
| .00	      			| Turn right ahead					 				|
| .00				    | Traffic signals      							|

And here is a plot of the predictions softmax probabilities:
![alt text][image5]
