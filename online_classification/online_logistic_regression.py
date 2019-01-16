import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# Pre-Processing : @SharmaNatasha 
#https://github.com/SharmaNatasha/Machine-Learning-using-Python/blob/master/Classification%20project/Spam_Detection.ipynb


# Removes features (words) that are not important
def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)


# Helper function for sigmoid function
def sigmoid(x_i, w) :
    z = np.exp(-np.dot(x_i, w.T))
    return (1 / ( 1 + z ))

# Cross entropy loss
def loss_fun(x_i, y, w) :
    sig_val = sigmoid(x_i, w)
    return  -(y*np.log(sig_val) + (1 - y)*np.log( 1 - sig_val))

# Total hindshight loss with arguements p as conditional probability and y  
def hindsight_total_loss(p, y) :
    n = y.shape[0]
    total_loss = 0
    for i in range(n): 
        total_loss += - ( y[i] * np.log(p[i,1]) + (1 - y[i]) * np.log(p[i,0]))
    return total_loss
        
# Returns gradient of loss function evaluated at w_vec
def gradient_update(w_vec, x_vec, y) :
    sig_val = sigmoid(x_vec, w_vec)
    #print("function value",sig_val)
    sig_prime = sig_val * (1 - sig_val)
    #print(sig_prime)
    z_t = ((1 - y) / (1 - sig_val) - y / sig_val) *sig_prime * x_vec
    #print("gradient value",z_t)
    #grad = (y * sig_val + (1 - y) * (1 - sig_val))*sig_prime * x_vec
    return z_t

# Performs online logistic regression
def online_log_regression( x, y, eta) :
    n, m = x.shape[0], x.shape[1]
    w = np.ones(m) * 0
    losses = np.zeros(n)
    cum_losses = np.zeros(n)
    classification = np.zeros(n)
    #eta = B/(G*(np.sqrt(2 * n)))

    for i in range(n) :
        x_vec = x[i,:].todense()
        loss_round_i = loss_fun(x_vec, y[i], w)
        classification[i] = (sigmoid(x_vec, w) >= 0.5).astype(int)
        losses[i] = loss_round_i
        cum_losses[i] =  cum_losses[i - 1] + loss_round_i
        z_t = gradient_update(w, x_vec, y[i])
        w = w - eta * z_t
        
    class_correct = 1 - np.sum(classification - y) / n
    return cum_losses , losses, w, class_correct, eta

# Underlying loss
def classification_error(w_hat, x, y) :
    n, m = x.shape[0], x.shape[1]
    prob = np.zeros(n)
    
    for i in range(n) :
        x_vec = x[i, :].todense()
        prob[i] = sigmoid(x_vec, w_hat)
    prob = (prob >= 0.5).astype(int)
    return ( 1 - np.sum(prob - y)/n)

cum_losses, losses, w, online_class, eta = online_log_regression(x, y, 1)

if __name__ == "main" :
    # Stopwords are considred (it, and, the, etc.)
    nltk.download('stopwords')
    message_data = pd.read_csv("spam.csv",encoding = "latin")
    message_data.head()

    message_data = message_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
    message_data = message_data.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})
    message_data.groupby('Spam/Not_Spam').describe()
    message_data_copy = message_data['message'].copy()

    # Two copies of the data with and without filtered words
    message_data_copy = message_data_copy.apply(text_preprocess)

    # Vectorize the data
    vectorizer1 = TfidfVectorizer("english")

    # Normalize
    message_mat = vectorizer1.fit_transform(message_data_copy)
    # 1 for no spam, 0 for spam (y)
    # x is dataset with each data point containing features (normalized) of relative frequencies of words

    y = np.array((message_data['Spam/Not_Spam'] == 'ham').astype(int))
    x = message_mat
    cum_losses, losses, w, online_class, eta = online_log_regression(x, y, 1)

    plt.plot(losses)
    plt.plot(cum_losses)
    plt.legend(('Round Loss', 'Cumulative Model Loss'))
    plt.show()

    print('Eta = ', eta)
    print("Correctly classified offline :",classification_error(w, x, y))
    print("Correctly classified online :", online_class)