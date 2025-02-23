import spacy
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')


"""
This chatbot program starts by loading the spacy model "English version", then uses it to vectorize the questions dataset 
as well as the user input(user's question), then it compares both of them to find the most similar question using cosine similarity, 
and returns the answer of the quesion that is the most similar to the user's question. 

the user can enter one question at a time, and after the chatbot returns an answer, the user can enter another question. 
finally, the user can also exit the chat by typing exit or bye or quit.
"""

# Load training JSON data from file
with open('questions.json', 'r') as file:
    training_data = json.load(file)

# Access questions
training_questions = [qa_pair['question'] for qa_pair in training_data['questions']]
training_answers = [qa_pair['answer'] for qa_pair in training_data['questions']]

nlp = spacy.load("en_core_web_md")

def feed_model(input_question): 
    doc = nlp(input_question)
    return doc


# vectorization of questions dataset using spacy model
# return the vectorized questions
questions_vectors = []
def vectorize_questions(training_questions):
    for question in training_questions:
        questions_vectors.append(feed_model(question))
    return questions_vectors

def get_most_similar_question(user_question):
    #convert user input and questions to vector using spaCy
    questions_to_vector = vectorize_questions(training_questions)
    input_to_vector = feed_model(user_question)
    # print(input_to_vector)
    similarity = []
    for question in questions_to_vector:
        similarity.append( input_to_vector.similarity(question))
    most_similar_index = np.argmax(similarity)

    # Retrieve the most similar question and its corresponding answer
    most_similar_question = training_questions[most_similar_index]
    most_similar_answer = training_answers[most_similar_index]
    return most_similar_question, most_similar_answer


def load_questions(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    questions = [qa_pair['question'] for qa_pair in data['questions']]
    answers = {qa_pair['question']: qa_pair['answer'] for qa_pair in data['questions']}
    return questions, answers


def load_greetings(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    greetings = {greeting['input'].lower(): greeting['response'] for greeting in data['greetings']}
    return greetings

questions, answers = load_questions('questions.json')
greetings = load_greetings('greetings.json')
answers.update(greetings)



end_chat = ["bye", "exit", "see you", "quit", "stop", "thank you", "thank you bye", "done"]
max_sequence_length = 50  


# Chatbot loop
print("Chatbot: Hi! I'm the chatbot. You can start the conversation by typing your message.")
while True:
    user_input = input("User: ")
    if user_input.lower() in end_chat:
        print("Chatbot: Goodbye!")
        break
    
    # Check if the input matches any greeting
    if user_input.lower() in answers:
        print("Chatbot:", answers[user_input.lower()])
        continue  # Skip further processing
    
    #call the function to find the most similar question
    #return the answer of that question
    similar_question = get_most_similar_question(user_input)
    most_similar_question, selected_answer = get_most_similar_question(user_input)
    print("Chatbot:", selected_answer)




