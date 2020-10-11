# FaceX
A system which find missing person with the face recognition system using One Shot Learning Method

# Abstract
High number of people goes missing everyday which includes children, Teenagers, Mentally Challenged, People with Alzheimer’s, etc .
On average, a child goes missing every 10 minutes in India, according to the women and child development ministry website for tracking missing children.
We have developed a system which will help to find missing people with help of face recognition system
https://www.hindustantimes.com/india-news/with-60-000-children-going-missing-in-india-every-year-social-media-has-propelled-child-lifting-fear/story-AvL4yvASeN4fgXQPoAkBKP.html

https://www.thehindu.com/news/national/maharashtra-records-highest-number-of-missing-women-ncrb/article30728296.ece

# Our System
When a person goes missing, the police can upload the picture of the person which will get stored in the database. When the public encounter a suspicious person, they can capture and upload the picture or video footage of public cameras into our portal. The face recognition model in our system will try to find a match in the database with the help of face encodings.

# Our Approach and How it is different from others
With normal deep learning method, model has to be trained on huge no. of labelled images of the employees and needs to be trained on large no. of epochs. This method may not be suitable because every time new employee comes in model needs to be trained.
Our approach is model is trained on fewer images of the People, but it can be used for newer People without retraining the model. This way of approach is called one shot learning.
One-shot learning is an object categorization problem in computer vision. Whereas most machine learning based object categorization algorithms require training on hundreds or thousands of images and very large datasets, one-shot learning aims to learn information about object categories from one, or only a few, training images.
So we have used FaceNet which was introduced by google in 2015 which uses one shot learning.

Main difference between FaceNet and other techniques is that it learns the mapping from the images and creates embeddings rather than using any bottleneck layer for recognition or verification tasks.

# Limitations
1)	Age of person plays an important role in finding missing person and if it was lost in early age than after span of years it becomes difficult to find due to growth of facial features.


# Future Scope 
1)	We are planning to extend by connecting in real-time to public-cameras.
2)	We have implemented on local host and with limited resources but try to host it on cloud for better result.

# Note
Web Integration is not available due to some error we will solve it in next commit

# How to Run 
1)In Data --> People --> Add folder with name of photo and its images

2)run prepare_data.py

3)run detect_face.py