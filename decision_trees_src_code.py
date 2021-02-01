from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_selection import mutual_info_classif
import random
import math
import graphviz
from collections import Counter

TRAINING_PORPORTION = 0.7
VALIDATION_PORPORTION = 0.15
TEST_PORPORTION = 0.15

def read_and_shuffle_labelled_data():
    """
    Read and shuffle data into [(headline, label)] format.
    """
    with open("clean_fake.txt") as fake_data_file:    
        fake_data = fake_data_file.readlines()    
    fake_data = [x.strip() for x in fake_data]
    
    fake_data_labelled = []
    for data in fake_data:
        fake_data_labelled.append((data, "fake_news"))

    with open("clean_real.txt") as real_data_file:    
        real_data = real_data_file.readlines()    
    real_data = [x.strip() for x in real_data]
    
    real_data_labelled = []
    for data in real_data:
        real_data_labelled.append((data, "real_news"))
    
    all_data = []
    all_data.extend(real_data_labelled)
    all_data.extend(fake_data_labelled)

    random.shuffle(all_data)
    return all_data

def load_data(vectorizer):
    labelled_data = read_and_shuffle_labelled_data()
  
    headlines = [] #All headlines from tuple in list
    for x in labelled_data:
        headlines.append(x[0])

    document_term_matrix = vectorizer.fit_transform(headlines) #fit and transform into document term matrix

    labels = [] #All corresponding labels from tuple in list
    for x in labelled_data:
        labels.append(x[1])

    rows = len(labelled_data) 
    endIndex_training = int(TRAINING_PORPORTION * rows) 
    startIndex_validation = endIndex_training 
    endIndex_validation = endIndex_training + int(VALIDATION_PORPORTION * rows) 
    startIndex_testing = int(TEST_PORPORTION * rows) + endIndex_training
    
    training_data = document_term_matrix[:endIndex_training]
    validation_data = document_term_matrix[startIndex_validation:endIndex_validation]
    test_data = document_term_matrix[startIndex_testing:]
                                                             
    train_labels = labels[:endIndex_training] 
    valid_labels = labels[startIndex_validation:endIndex_validation]
    test_labels =  labels[startIndex_testing:]

    return (training_data, train_labels), (validation_data, valid_labels), (test_data, test_labels)


def evaluate_training_model(training_model, validation_data, validation_labels): 
    #evaluates the performance of each one on the validation set
    
    model_predictions = training_model.predict(validation_data)
    
    correct_count = 0 #Initialize counter of number of correct predictions
    total_count = len(validation_labels) #Number of total counts
    
    for i in range(total_count):
        if model_predictions[i] == validation_labels[i]:
            correct_count += 1
    score = correct_count / total_count

    return score #proportion of correct_count / total_count



def select_model(training_data, training_labels, validation_data, validation_labels):
    # Select best model/decision tree with highest validaiton score    
    best_model = None #Initialize best model/tree
    best_score = 0 #Initialize best score
    
    for default_criterion in ["entropy", "gini"]: 
        for max_depth in range(1, 6): #five different values from max depth {1,2,3,4,5}
            model = DecisionTreeClassifier(criterion=default_criterion, max_depth=max_depth)
            model.fit(training_data, training_labels)  #train and fit model on the validation data
            score = evaluate_training_model(model, validation_data, validation_labels) #calls helper to evaluate highest score
            print("Tree with depth "  + str(max_depth) + " and criteria = " + str(default_criterion)+
                  " : validation score = " + str(score))
            if score > best_score: #update best score 
                best_score = score
                best_model = model
                       
    print("\n"+ "Best model is:")
    print(best_model)
    
    print("\n"+ "Best score is:")
    print(best_score)
    return best_model

def visualize_best_tree(training_model, tree_name):
    export_graphviz(training_model, out_file=tree_name+".dot", max_depth=2,
                    class_names=training_model.classes_,
                    feature_names=vectorizer.get_feature_names())


def log_entropy(n):
    #Compute n * log(n) since entropy H(x) = -Summation(P_n*log(P_n))
    if n == 0:
        return 0
    return n * math.log(n)

def compute_entropy(labels_list):
    # Compute entropy of given labels
    num_total = len(labels_list)
    num_fake = 0
    num_real = 0 
    for x in labels_list:
        if x == "fake_news":
            num_fake +=1
        else:
            num_real +=1
            
    total_fake = log_entropy(num_fake / num_total)
    total_real = log_entropy(num_real / num_total)
    total = -(total_fake + total_real) #By entropy formula H(x) = -plog(p) - qlog(q)
    return total


def compute_entropy_for_specific_vocab(document_term_matrix, labels, vocabulary, specific_vocab):
    # Column of the word in the document matrix
    #print(document_term_matrix)
    array_of_specific_vocab = document_term_matrix.getcol(vocabulary[specific_vocab])
    word_columns = array_of_specific_vocab.toarray().flatten()

    present_indices = {0: (1 - word_columns).nonzero()[0],  #index of vocab not in column
        1: word_columns.nonzero()[0]}  # index of vocab in column

    num_labels = len(labels)

    presented_labels_count = Counter( #store counts for corresponding fake and real news labels
            (presence, labels[ind]) for presence in present_indices for ind in present_indices[presence])

    probability_presented_labels = { #convert presented_labels_count into corresponding probabilites 0 <=p<=1
        x: (presented_labels_count[x] / num_labels) 
        for x in presented_labels_count}

    probability_presented_vocab = { # Probabilities of presented vocab words
        presence: len(present_indices[presence]) / num_labels
        for presence in present_indices}
    

    cond_probability_label_given_presented_vocab = { # Conditional probabilities of label given presented vocab word
        (presence, label): probability_presented_labels[(presence, label)] /
                           probability_presented_vocab[presence]
        for (presence, label) in probability_presented_labels}
    
    return -1 * sum( #compute entropy using formula P(T) * nlogn(P(T|X)) 
        probability_presented_vocab[presence] * log_entropy(cond_probability_label_given_presented_vocab[(presence, label)])
        for (presence, label) in cond_probability_label_given_presented_vocab)


def compute_information_gain(document_term_matrix, headline_labels, vectorized_vocabulary, specific_vocab):
    #Compute info gain by formula: Gain(T,X) = Entropy(T) - Entropy(T|X)
    return compute_entropy(headline_labels) - compute_entropy_for_specific_vocab(document_term_matrix, headline_labels, 
                          vectorized_vocabulary, specific_vocab)

if __name__ == '__main__':
    vectorizer = CountVectorizer(analyzer='word',min_df=3,binary=True)
    (training_data, training_labels), (validation_data, validation_labels), (test_data, test_labels) = load_data(vectorizer)
    # Graphviz output best tree diagram
    best_model = select_model(training_data, training_labels, validation_data, validation_labels)
    visualize_best_tree(best_model, "Top_Tree")
    print("\n")

    # Compute information gain on specific vaocabulary words
    word_labels_tuple = [] #initialize list of tuples (vocab, info_gain) for sorting purposes later
    for specific_vocab in ["donald", "trump", "hillary", "clinton", "election", "putin", "china", "korea", "america", "trade"]:
        information_gain = compute_information_gain(training_data, training_labels, vectorizer.vocabulary_, specific_vocab)
        word_labels_tuple.append((specific_vocab, information_gain))
    
    #Print and Sort Information Gain in descending order
    print("Sorted Information Gain by specific words in descending order: ")
    sorted_labels = sorted(word_labels_tuple,key=lambda x: x[1], reverse=True) 
    for item in sorted_labels:
        print(f"Information Gain By splitting on " + item[0] +" : " + str(item[1]))
        
        
        
        
        
        
        
        
        
        
        
        
        
        