
#pip install tensorflow
#pip install pydot
#pip install pydotplus
#pip install graphviz
#pip install pillow
import os
import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import tensorflow as tf
##pip install keras
##pip install keras==2.1.5
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
# small library for seeing the progress of loops.
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()
import pydot
import graphviz
from tensorflow.keras.utils import plot_model

#tf.config.run_functions_eagerly(True)


# function - loading text file.

def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# # Getting all their images with captions


def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions ={}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [ caption ]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions


# # Data Cleaning - lower case, removing punctuations and digits


def cleaning_text(captions):
    table = str.maketrans('','',string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):
            img_caption.replace("-"," ")
            desc = img_caption.split()
            #converts to lowercase
            desc = [word.lower() for word in desc]
            #remove punctuation from each token
            desc = [word.translate(table) for word in desc]
            #remove hanging 's and a 
            desc = [word for word in desc if(len(word)>1)]
            #remove tokens with numbers in them
            desc = [word for word in desc if(word.isalpha())]
            #convert back to string
            img_caption = ' '.join(desc)
            captions[img][i]= img_caption
    return captions


def text_vocabulary(descriptions):
    # build vocabulary of all unique words
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab


# # All descriptions in one file


def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc )
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()


# # Defining paths for text and image folders

os.chdir(r"C:/Users/akash/Desktop/S3/proj")

print("Current working directory is:", os.getcwd())

dataset_text = "Flickr8k_text"

dataset_images = "Flicker8k_Dataset"


# # Preparing our text data


filename = dataset_text + "/" + "Flickr8k.token.txt"


# Loading and mapping into descriptions dictionary


descriptions = all_img_captions(filename)

print("Length of descriptions =" ,len(descriptions))


# Cleaning the descriptions using 'clean_descriptions' function defined


clean_descriptions = cleaning_text(descriptions)


# building a vocabulary using these clean description using the 'text_vocabulary' fucntion defined

vocabulary = text_vocabulary(clean_descriptions)

print("Length of vocabulary = ", len(vocabulary))


# Saving descriptions to offline file



save_descriptions(clean_descriptions, "descriptions2.txt")


# # Working with the images

# defining feature extracting function


def extract_features(directory):
        model = Xception( include_top=False, pooling='avg' )
        features = {}
        for img in tqdm(os.listdir(directory)):
            filename = directory + "/" + img
            image = Image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            #image = preprocess_input(image)
            image = image/127.5
            image = image - 1.0
            feature = model.predict(image)
            features[img] = feature
        return features


# Applying feature extractor on the images


features = extract_features(dataset_images)


# Saving the pickle file offline


dump(features, open("features2.p","wb"))


# Loading the previously saved pickle file


features = load(open("features2.p","rb"))


# Defining function to load the images, clean descriptions and features.


def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos


def load_clean_descriptions(filename, photos): 
    #loading clean_descriptions
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):

        words = line.split()
        if len(words)<1 :
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)

    return descriptions


def load_features(photos):
    #loading all features
    all_features = load(open("features2.p","rb"))
    #selecting only needed features
    features = {k:all_features[k] for k in photos}
    return features



filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"



train_imgs = load_photos(filename)

train_descriptions = load_clean_descriptions("descriptions2.txt", train_imgs)

train_features = load_features(train_imgs)


# Function to convert dictionary to clean list of descriptions


def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# Creating tokenizer class, this will vectorise text corpus, each integer will represent token in dictionary


def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer


# Applying tokenizer and saving the file offline


tokenizer = create_tokenizer(train_descriptions)

dump(tokenizer, open('tokenizer2.p', 'wb'))

vocab_size = len(tokenizer.word_index) + 1

vocab_size


# Calculate maximum length of captions


def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)
    
max_length = max_length(descriptions)

max_length


# Data generator and creating sequences


def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            #retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield ([input_image, input_sequence], output_word)

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


# Checking the shape of inputs(a,b) and output c.


[a,b],c = next(data_generator(train_descriptions, features, tokenizer, max_length))

a.shape, b.shape, c.shape


# Defining the caption model functions


def define_model(vocab_size, max_length):
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# # Training the model


print('Dataset: ', len(train_imgs))

print('Descriptions: train=', len(train_descriptions))

print('Photos: train=', len(train_features))

print('Vocabulary Size:', vocab_size)

print('Description Length: ', max_length)


# Defining model parameters and saving models into a directory 'models'


model = define_model(vocab_size, max_length)

epochs = 10

steps = len(train_descriptions)

os.mkdir("models")



for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit_generator(generator, epochs=1, steps_per_epoch= steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")




