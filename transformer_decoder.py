import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import cv2
import os
import PyPDF2
from docx import Document
import keras
reverse_vocab = {i: f"word_{i}" for i in range(10000)}  # Adjust size as necessary

# Placeholder function to preprocess input text
def process_input_text(input_text):
    # Implement this to convert input_text to the format your model expects
    encoded_input = np.random.randint(0, len(reverse_vocab), (1, max_len))
    predicted_sequence = decoder_model.predict(encoded_input)[0]
    return predicted_sequence

# Function to decode a sequence of indices into readable text
def decode_sequence(sequence):
    return ' '.join(reverse_vocab.get(index, '[UNK]') for index in sequence)

# Placeholder function to generate a response using the model
def generate_response(model, text_input):
    text_input = np.array([text_input])  # Ensure text input is in array form
    response_indices = process_input_text(text_input[0])
    response_text = decode_sequence(response_indices)
    return response_text
# Function to capture video frame (dummy function for now)
def capture_video_frame():
    return np.zeros((225, 225, 3), dtype=np.float32)  # Dummy video input

# Initialize the Tkinter root
root = tk.Tk()
root.title("Chatbot")

# Create the conversation display
conversation_frame = tk.Frame(root)
conversation_frame.pack(pady=10)

conversation_list = tk.Listbox(conversation_frame, width=80, height=20)
conversation_list.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar = tk.Scrollbar(conversation_frame, orient=tk.VERTICAL)
scrollbar.config(command=conversation_list.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

conversation_list.config(yscrollcommand=scrollbar.set)

# Function to handle user input and generate response
def send_message(event=None):
    user_input = user_entry.get()
    if user_input:
        conversation_list.insert(tk.END, f"You: {user_input}")
        user_entry.delete(0, tk.END)
        
        # Process the input and generate response using the custom model
        processed_input = process_input_text(user_input)
        
        # Capture video frame (dummy function for now)
        video_frame = capture_video_frame()
        
        response_text = generate_response(decoder_model, processed_input)
        
        conversation_list.insert(tk.END, f"Oni: {response_text}")
        conversation_list.yview(tk.END)  # Scroll to the bottom

# Create the entry widget for user input
user_entry = tk.Entry(root, width=80)
user_entry.pack(pady=10)
user_entry.bind("<Return>", send_message)

# Create the send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

# Functions to read PDF and DOCX files
def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            content = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                content += page.extract_text()
            return content
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return None

def read_docx(file_path):
    try:
        doc = Document(file_path)
        content = ""
        for para in doc.paragraphs:
            content += para.text + "\n"
        return content
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return None

# Function to handle file upload
def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        conversation_list.insert(tk.END, f"File uploaded: {file_path}")
        if file_path.endswith('.pdf'):
            data = read_pdf(file_path)
        elif file_path.endswith('.docx'):
            data = read_docx(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = file.read()
        
        if data:
            conversation_list.insert(tk.END, f"File content: {data[:100]}...")  # Display first 100 characters
            processed_input = process_input_text(data)
            response_text = generate_response(decoder_model, processed_input)
            conversation_list.insert(tk.END, f"Bot: {response_text}")

# Create the file upload button
upload_button = tk.Button(root, text="Upload File", command=upload_file)
upload_button.pack()

# Training loop to use files from ./knowledge_base
def training_loop(model, knowledge_base_dir):
    for filename in os.listdir(knowledge_base_dir):
        file_path = os.path.join(knowledge_base_dir, filename)
        if os.path.isfile(file_path):
            try:
                if file_path.endswith('.pdf'):
                    data = read_pdf(file_path)
                elif file_path.endswith('.docx'):
                    data = read_docx(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = file.read()
                
                if data:
                    processed_input = process_input_text(data)
                    # Example training call
                    # model.train_on_batch(processed_input, target_output)  # Adjust based on your model's training method
            except Exception as e:
                print(f"Error processing file: {file_path}: {e}")

# Path to the knowledge base directory
knowledge_base_dir = 'C:/Users/jonny/Documents/PATH/ONI/knowledge_base/remembered_texts/'

# Define the vocabulary size
vocab_size = 600000

# Define the embedding dimension
embedding_dim = 128

def create_decoder_model(max_len):
    # Input layer for the decoder input
    decoder_inputs = keras.Input(shape=(max_len,))
    decoder_embeddings = keras.layers.Embedding(vocab_size, embedding_dim)(decoder_inputs)

    # LSTM layer with return_sequences=True for recurrent generation
    decoder_lstm = keras.layers.LSTM(256, return_sequences=True)(decoder_embeddings)

    # Output layer with softmax for probability distribution
    decoder_outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder_lstm)

    # Model definition
    model = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)
    
    # Compile the model (optimizer and loss function specific to your task)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return model

max_len = 500

# Create the model
decoder_model = create_decoder_model(max_len)

# Dummy encoded input for demonstration purposes
encoded_input = np.random.randint(0, vocab_size, (1, max_len))

# Predict the sequence
predicted_sequence = decoder_model.predict(encoded_input)[0]

# Decode the predicted sequence (optional)
# This typically involves mapping indices back to words in your vocabulary
# Assuming you have a function `decode_sequence` to handle this
# decoded_sequence = decode_sequence(predicted_sequence)

# For demonstration, print the predicted sequence (as indices)
print(predicted_sequence)

# Start the training loop
training_loop(decoder_model, knowledge_base_dir)

# Run the Tkinter main loop
#root.mainloop()

