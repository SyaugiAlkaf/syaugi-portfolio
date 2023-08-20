import json
import random

# Load the dataset from the JSON file
with open('train_light.json', 'r') as json_file:
    dataset = json.load(json_file)

# Create a dictionary to map question IDs to answers
question_answers = {}
for entry in dataset:
    annotations = entry.get('annotations', [])
    for annotation in annotations:
        if annotation['type'] == 'singleAnswer':
            question_answers[entry['id']] = annotation['answer'][0]

# Define the chatbot function
def chatbot(question):
    for entry in dataset:
        if entry['question'] == question:
            question_id = entry['id']
            if question_id in question_answers:
                return question_answers[question_id]
            else:
                qa_pairs = entry['annotations'][0]['qaPairs']
                for qa_pair in qa_pairs:
                    if qa_pair['question'] == question:
                        return random.choice(qa_pair['answer'])
    return "I'm sorry, I don't have an answer for that question."

# Main loop for interacting with the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = chatbot(user_input)
    print("Chatbot:", response)
