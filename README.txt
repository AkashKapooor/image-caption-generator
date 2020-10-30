Image caption generator on the flickr8k dataset of images.

Python based project using deep learning techniques of CNN and RNN together.

The image features are extracted using the latest 'Xception model' which is a CNN model trained on the imagenet dataset and then we feed the extracted features into the LSTM model which is a part of RNN which will be responsible for generating the image captions.

Apart from the images dataset, a very important part of this study is the token file which containes image name and their respective captions separated by \n which will be very useful in training our model.

Images - https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

Text - https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip


Files - 

requirements.txt - set and versions of libraries and packages used for this project.

Code_1.2_Training - feature extraction of images dataset, text cleaning of text dataset, generating tokenizer, descriptions and features file and training models and validating the model using BLEU score

Testing_gen_model - Codes to test the generated models on the images dataset.

descriptions2.txt - Cleaned out and processed descriptions.

features2.p - pickle file containing extracted features of images using Xception model

tokenizer2.p - the tokenizer file.

All files available here - 

https://drive.google.com/drive/folders/1wYElDJx64FSxyu4dJzD-znLvIhwRoWkn?usp=sharing



Please Note- In requirements.txt, use keras==2.1.5 only if you're using an older version of tensorflow. (v2.3.1 includes the keras package)
