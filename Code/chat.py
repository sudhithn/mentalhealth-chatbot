# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Developer details: 
    # Names: Sudhith N
    # Role: Architects
    # Code ownership rights: Sudhith N
# Version:
    # Version: V 1.1 (26 Nov 2024)
        # Developers: Sudhith N
        # Unit test: Pass
        # Integration test: Pass
 
# Description: This code implements a graphical user interface for chatbot interactions
    # Framework: TensorFlow, Tkinter
    # ML Components: Neural Network Inference
    # Data Processing: Text Processing, Intent Prediction
    # Features:
        # GUI Interface: Yes
        # Real-time Chat: Yes
        # Model Inference: Yes
        # Response Generation: Yes
        # Chat History: Yes

# Dependencies: 
    # Environment:     
        # Python 3.12.6
        # TensorFlow 2.18.0
        # NLTK 3.9.1
        # NumPy 2.0.2

# Import Libraries
import tkinter as tk                   # GUI framework for creating interface
from tkinter import scrolledtext       # Scrollable text widget for chat history
from tkinter import messagebox         # Message dialog boxes for notifications
import json                           # For reading intent configuration files
import random                         # For random response selection
import numpy as np                    # For numerical operations and arrays
import pickle                         # For loading serialized model data
import nltk                          # Natural language processing toolkit
import os                            # File and path operations
from nltk.stem import WordNetLemmatizer  # Word lemmatization for text processing
import tensorflow as tf               # Deep learning framework for model inference
from pathlib import Path             # Object-oriented filesystem paths

# Get the project root directory
BASE_DIR = Path(__file__).parent.parent

# Define directory paths
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models"

# Define file paths
INTENTS_FILE = DATA_DIR / "intents.json"
MODEL_FILE = MODELS_DIR / "chatbot_model.h5"
WORDS_FILE = MODELS_DIR / "words.pkl"
CLASSES_FILE = MODELS_DIR / "classes.pkl"

def validate_paths():
    """
    Validate that all required files exist before starting the chat interface
    
    Raises:
        FileNotFoundError: If any required file is missing
    """
    required_files = {
        "Intents File": INTENTS_FILE,
        "Model File": MODEL_FILE,
        "Words File": WORDS_FILE,
        "Classes File": CLASSES_FILE
    }
    
    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at: {path}\n"
                f"Please ensure all required files are present and run bot.py first to generate model files."
            )

class ChatbotInterface:
    """
    Manages the chatbot's graphical interface and response generation
    
    Implementation:
        - Creates and manages GUI elements
        - Handles user input processing
        - Manages model inference
        - Generates appropriate responses
        
    Attributes:
        lemmatizer (WordNetLemmatizer): Word lemmatization tool
        model (tf.keras.Model): Loaded neural network model
        intents (dict): Loaded intent patterns and responses
        words (list): Processed vocabulary
        classes (list): Intent categories
    """
    
    def __init__(self):
        """
        Initialize the chatbot interface
        
        Implementation:
            - Downloads required NLTK data
            - Loads model and preprocessing data
            - Initializes GUI components
            
        Processing Steps:
            1. Setup NLTK requirements
            2. Load model and data
            3. Initialize GUI
            4. Configure event handlers
        """
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('wordnet')

        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        self.load_model_data()
        self.setup_gui()

    def load_model_data(self):
        """
        Load trained model and associated data
        
        Implementation:
            - Loads saved model from file
            - Loads preprocessed words and classes
            - Loads intent configurations
            
        Error Handling:
            - Handles missing files
            - Validates loaded data
            - Reports loading errors
        """
        try:
            with open(WORDS_FILE, 'rb') as f:
                self.words = pickle.load(f)
            with open(CLASSES_FILE, 'rb') as f:
                self.classes = pickle.load(f)
            self.model = tf.keras.models.load_model(MODEL_FILE)
            
            with open(INTENTS_FILE, 'r') as f:
                self.intents = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model data: {str(e)}\n"
                               "Please ensure bot.py has been run first to generate the model files.")
            raise

    def setup_gui(self):
        """
        Create and configure GUI elements
        
        Implementation:
            - Creates main window
            - Adds chat display area
            - Creates input field
            - Adds control buttons
            
        Layout:
            - Chat history at top
            - Input field below
            - Send button at bottom
        """
        self.root = tk.Tk()
        self.root.title("Mental Health Chatbot")
        
        # Create chat frame
        chat_frame = tk.Frame(self.root)
        chat_frame.pack(pady=10)
        
        # Create chat display
        self.chat_box = scrolledtext.ScrolledText(chat_frame, width=50, height=20, wrap=tk.WORD)
        self.chat_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_box.config(state=tk.DISABLED)
        
        # Create input field
        self.entry = tk.Entry(self.root, width=40)
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", lambda e: self.handle_send())
        
        # Create send button
        send_button = tk.Button(self.root, text="Send", command=self.handle_send)
        send_button.pack()

    def clean_up_sentence(self, sentence):
        """
        Preprocess input sentence
        
        Implementation:
            - Tokenizes sentence
            - Lemmatizes words
            - Converts to lowercase
            
        Args:
            sentence (str): Input text
            
        Returns:
            list: Processed words
        """
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def create_bow(self, sentence):
        """
        Create bag-of-words representation
        
        Implementation:
            - Processes sentence
            - Creates binary word vector
            - Matches against vocabulary
            
        Args:
            sentence (str): Input text
            
        Returns:
            numpy.ndarray: Bag-of-words vector
        """
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        
        for word in sentence_words:
            for i, w in enumerate(self.words):
                if w == word:
                    bag[i] = 1
                    
        return np.array(bag)

    def predict_class(self, sentence):
        """
        Predict intent class for input
        
        Implementation:
            - Creates input representation
            - Performs model inference
            - Processes prediction results
            
        Args:
            sentence (str): Input text
            
        Returns:
            list: Predicted intents and probabilities
        """
        bow = self.create_bow(sentence)
        res = self.model.predict(np.array([bow]))[0]
        
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = [{"intent": self.classes[r[0]], "probability": str(r[1])} 
                      for r in results]
        return return_list

    def get_response(self, intent_list):
        """
        Generate response based on predicted intent
        
        Implementation:
            - Finds matching intent
            - Selects random response
            - Handles fallback responses
            
        Args:
            intent_list (list): Predicted intents
            
        Returns:
            str: Selected response
        """
        if not intent_list:
            return "I'm not sure I understand. Could you rephrase that?"
            
        tag = intent_list[0]['intent']
        list_of_intents = self.intents['intents']
        
        for intent in list_of_intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

    def handle_send(self):
        """
        Process send button click
        
        Implementation:
            - Gets user input
            - Processes message
            - Updates chat display
            - Handles quit command
            
        Processing Steps:
            1. Get user input
            2. Generate response
            3. Update display
            4. Clear input field
        """
        msg = self.entry.get()
        if msg.lower() == 'quit':
            self.root.destroy()
            return
            
        if msg:
            self.chat_box.config(state=tk.NORMAL)
            self.chat_box.insert(tk.END, f"You: {msg}\n")
            response = self.get_response(self.predict_class(msg))
            self.chat_box.insert(tk.END, f"Bot: {response}\n\n")
            self.chat_box.config(state=tk.DISABLED)
            self.chat_box.see(tk.END)
            self.entry.delete(0, tk.END)

    def run(self):
        """
        Start the chat interface
        
        Implementation:
            - Initializes main loop
            - Handles window events
            - Manages application lifecycle
        """
        self.root.mainloop()

def main():
    """
    Main execution function
    
    Implementation:
        - Creates interface instance
        - Handles initialization errors
        - Starts application
    """
    try:
        # Validate paths before starting
        validate_paths()
        
        chatbot = ChatbotInterface()
        chatbot.run()
    except Exception as e:
        messagebox.showerror("Error", f"Application error: {str(e)}")

if __name__ == "__main__":
    main()