import spacy
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import textwrap


'''
To run the testing code, type 'python testing_chatbot.py' in terminal
This code test the chatbot performance on the questions that are in the validation set (validation.json) which are different from the ones in the questions dataset
that the chatbot draws answers from, however, they have the same meaning.

the output will be an example of a user question, and they the bot finds a similar question in the questions dataset.
finally to check if the bot got it correctly, it compares it to the answers in the validation set that are assigned to each question, 
which the answer is the most similar question that the bot should find.

the code the prints the user question, followed by the most similar question that it found, and the score for the number of questions it got right.
'''
# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load training JSON data from file
with open('questions.json', 'r') as file:
    training_data = json.load(file)

# Access training questions and answers
training_questions = [qa_pair['question'] for qa_pair in training_data['questions']]
training_answers = [qa_pair['answer'] for qa_pair in training_data['questions']]

nlp = spacy.load("en_core_web_md")

def feed_model(input_question): 
    doc = nlp(input_question)
    return doc


# vectorization of questions dataset using spacy model
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
validation_questions, validation_answers = load_questions('validation.json')
answers.update(greetings)

with open('validation.json', 'r') as file:
    data = json.load(file)

# Extract the answers
validation_answerss = [entry['answer'] for entry in data['questions']]

end_chat = ["bye", "exit", "see you", "quit", "stop", "thank you", "thank you bye"]
max_sequence_length = 50  # Adjust as needed
# Chatbot loop

print("Chatbot: Hi! I'm the chatbot. You can start the conversation by typing your message.")

i = -1
score = 0
for question in validation_questions:
    i += 1
    user_input = validation_questions[i]
    if user_input.lower() in end_chat:
        print("Chatbot: Goodbye!")
        break

    similar_question = get_most_similar_question(user_input)
    most_similar_question, selected_answer = get_most_similar_question(user_input)
    if most_similar_question in validation_answerss:
        score += 1
        str(score)
    user_input_formatted = "{:<45}".format(user_input)
    print("User Question: --> ", user_input_formatted, "   ", "----- Most Similar Question Found: --> ", most_similar_question, sep='')
print("----------------------------")
print()
print("Overall score on validation data: ", score ,"/10")
print()
