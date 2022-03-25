# NLP-use-case-2
NLP-use-case-2

in this task we will be making a hybrid chatbot 

plan

1. first we want to get the intents which we will be using to train the bot the link to which is link 1 

2. the second thing we want to do is setup gpu processing, as i am on a windows mechine i will be going through the processfor that. first we want to download cuda and cudnn for nividia GPUs links 2 and 3 (make sure to install the newest versions orthey may not be compatible with tensorflow) and add the bins to your enviroment paths 

3. next we want to install the packages we will need for this prooject namely tensorflow, keras, pickle, nltk, numpty and random using pip install or other package managers

4.  now we are ready to start training our chatbot so we ned to set up our training class by importing os and redirecting the dictionary look up to the cuda bin, import our intents as well as importing all the packages we installed as shown in train_chatbot.py

5. next we want to preprocess our training data
first we want to tokenize our patterns in our intents which will be the words the bot uses as a key that the bot uses to determine intentions so its these words that the programme will look for in the user imput to develope a response. we will do this via nltk tokenize
next we want to remove duplicate words and ignore punctuation and set words to lower case using lamentize 
lastly we are left with a words variable which has all the words that has gone through the preprocessing and a classes variable that holds all the tags which are the catagories that input snf responses come under and using pickle dump to store these values as files that can be used to make predictions (link 5)

6. now we are ready to create our training and test data and convert the words into a series of 1 and 0s to indilcate if the word in that position of the documents is present
we start by making an empty output variable that will just contain a list of 0s the length of the classes variable
next we want to make the bag of words for each of the patterens in the intents this will show a 1 for each word that is in the pattern for its position in the words variable
then we do a similar thing for the tags using classes
then we set these values in a training variable which we will use to train the data

7. now we are finaly ready to build our model using sequential and sgd from keras (6,7)
first we sttart making our sequential model with our first layer, the input shape being the first value for which we use the first pattern 

after this we want to compile our model
first we set up our sgd or stochastic gradient descent thsi basiccly deterrmines the learning rate of our model which is important as if its set to low the modle will take too long to run and also be overtrained but if its too high it will be undertrained (8), we will be using nesterovs accelerated gradient that givves more wieght to the learning rate and the current gradient on each training cycle than if we just use momentum based gradient
then using this we compile our model using catagorical cross entropy loss (9)
lastly we want to make a histogram of the model and save the model and histogram to a file so it can be used without having to run this training every time 

8. lastly we have to make our GUI. this part consists of three things:
1. creating our GUI that can take a user input
2. taking that user input and transforming it to get the usefull information out and turning it into a form that our chatbot can understand 
3. getting a responnse out of the model and displaying it to the user

first of we do all our imports which are mostly importing the data we created in the training section plus a couple others most notably tkinter which we will be using to create the GUI itself
next we want to make a clean up class that basicly just takes the user input and tokenizes and turns it to lower case 
next we want to make a bag of words function that first runs the cleaning then turns taht list of words into another series of 1s and 0s to indicate which words in the words variable are present
then we take this bag of words and run it through are model to give us corrolation value to each of the tags setting an error threshhold to remove responses under that set corrolation and ordering the responses from most to least correloted 
then taking the intent with the highest correlation we get a random response from it and return it to the user
lastly we create the gui using tkinter designing a chat window that users can type in and upon pressing the send button starts the process of taking their input and getting back a response

and thats it we are done, i hopw this has all made sense the links are their to give a fuller explination to the more important packages and functions we used during the project.








links:
1. katana asssistant- intents
https://github.com/katanaml/katana-assistant/blob/master/mlbackend/intents.json

2. cuda toolkit
https://developer.nvidia.com/cuda-toolkit

3. nvidia cudnn
https://developer.nvidia.com/cudnn

2. Random documentation
https://docs.python.org/3/library/random.html

3. keras documentation
https://keras.io/

4. nltk documentation
https://www.nltk.org/

5. pickle documentation
https://docs.python.org/3/library/pickle.html

6. keras sequential
https://keras.io/guides/sequential_model/

7. keras sgd
https://keras.io/api/optimizers/sgd/

8. understanding learning rate
https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/

9. understanding catagorical cross entropy loss
https://vitalflux.com/keras-categorical-cross-entropy-loss-function/

10. Nesterov's accelerated gradient descent
https://stats.stackexchange.com/questions/179915/whats-the-difference-between-momentum-based-gradient-descent-and-nesterovs-acc