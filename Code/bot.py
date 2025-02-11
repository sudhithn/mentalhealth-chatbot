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
 
# Description: This code implements a chatbot training system using neural networks
    # Framework: TensorFlow
    # ML Components: Neural Networks, NLP Processing
    # Data Processing: Text Tokenization, Intent Classification
    # Features:
        # Data Preprocessing: Yes
        # Model Training: Yes
        # Intent Classification: Yes
        # Word Embeddings: Yes
        # Model Persistence: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependencies: 
    # Environment:     
        # Python Python 3.12.6
        # TensorFlow 2.18.0
        # NLTK 3.9.1
        # NumPy 2.0.2

# Import Libraries
import random                          # For random shuffling and selection operations
import json                           # For reading and parsing JSON intent files
import pickle                         # For serializing and deserializing Python objects
import numpy as np                    # For numerical operations and array handling
import tensorflow as tf               # Deep learning framework for model creation and training
import nltk                          # Natural Language Toolkit for text processing
from nltk.stem import WordNetLemmatizer  # For word lemmatization
import os                            # For operating system dependent functionality
from pathlib import Path             # For platform-independent path handling


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
    Validate that all required files and directories exist
    
    Raises:
        FileNotFoundError: If any required path is missing
        
    Note:
        Creates Models directory if it doesn't exist
    """
    # Check Data directory and intents file
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found at: {DATA_DIR}\n"
            "Please ensure the project directory structure is maintained"
        )
    
    if not INTENTS_FILE.exists():
        raise FileNotFoundError(
            f"Intents file not found at: {INTENTS_FILE}\n"
            "Please ensure intents.json is present in the Data directory"
        )
    
    # Create Models directory if it doesn't exist
    MODELS_DIR.mkdir(exist_ok=True)

# Download required NLTK data components
nltk.download('punkt')               # Tokenizer for splitting text into sentences
nltk.download('wordnet')            # Lexical database for lemmatization
nltk.download('punkt_tab')          # Additional tokenization data
nltk.download('averaged_perceptron_tagger')  # For part-of-speech tagging
nltk.download('omw-1.4')            # Open Multilingual Wordnet

class ChatbotTrainer:
    """
    Manages the training process for the chatbot neural network
    
    Implementation:
        - Processes training data from intents file
        - Creates and trains neural network model
        - Handles data preprocessing and tokenization
        - Manages model persistence
        
    Attributes:
        lemmatizer (WordNetLemmatizer): Word lemmatization tool
        intents (dict): Loaded intent patterns and responses
        words (list): Processed vocabulary
        classes (list): Intent categories
        documents (list): Training document pairs
    """
    
    def __init__(self):
        """
        Initialize the chatbot trainer
        
        Implementation:
            - Sets up lemmatizer
            - Loads intent data
            - Initializes training variables
            
        Processing Steps:
            1. Initialize lemmatizer
            2. Load intents data
            3. Set up training containers
        """
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.intents = json.loads(INTENTS_FILE.read_text())
        except Exception as e:
            raise Exception(f"Error loading intents file: {str(e)}")
            
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_chars = ['?', '!', '.', ',']

    def preprocess_data(self):
        """
        Process training data from intents
        
        Implementation:
            - Tokenizes patterns
            - Performs lemmatization
            - Creates word and class lists
            - Builds training documents
            
        Processing Steps:
            1. Extract patterns and tags
            2. Tokenize and lemmatize words
            3. Build vocabulary and class lists
            4. Create document pairs
        """
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(word.lower()) 
                     for word in self.words if word not in self.ignore_chars]
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))

        # Save processed data
        try:
            with open(WORDS_FILE, 'wb') as f:
                pickle.dump(self.words, f)
            with open(CLASSES_FILE, 'wb') as f:
                pickle.dump(self.classes, f)
        except Exception as e:
            raise Exception(f"Error saving preprocessed data: {str(e)}")

    def create_training_data(self):
        """
        Create training data for neural network
        
        Implementation:
            - Creates bag-of-words representations
            - Builds output vectors
            - Prepares final training set
            
        Returns:
            tuple: Training features and labels
            
        Processing Steps:
            1. Initialize training list
            2. Create bag-of-words for each document
            3. Build output rows
            4. Convert to numpy arrays
        """
        training = []
        output_empty = [0] * len(self.classes)

        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append(bag + output_row)

        random.shuffle(training)
        training = np.array(training)

        train_x = training[:, :len(self.words)]
        train_y = training[:, len(self.words):]

        return train_x, train_y

    def build_model(self, train_x, train_y):
        """
        Create and train the neural network model
        
        Implementation:
            - Defines model architecture
            - Configures training parameters
            - Trains the model
            - Saves trained model
            
        Args:
            train_x (numpy.ndarray): Training features
            train_y (numpy.ndarray): Training labels
            
        Processing Steps:
            1. Define neural network layers
            2. Compile model with optimizer
            3. Train model on data
            4. Evaluate and save model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
        ])

        adam = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', 
                     optimizer=adam, 
                     metrics=['accuracy'])

        print("Training model...")
        model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

        loss, accuracy = model.evaluate(train_x, train_y, verbose=0)
        print(f'Training accuracy: {accuracy * 100:.2f}%')

        try:
            model.save(MODEL_FILE)
            print(f'Model training completed and saved to {MODEL_FILE}!')
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")

def main():
    """
    Main execution function for training process
    
    Implementation:
        - Orchestrates training workflow
        - Handles exceptions
        - Reports progress
    """
    try:
        # Validate paths before starting
        validate_paths()
        
        # Initialize and train the chatbot
        trainer = ChatbotTrainer()
        trainer.preprocess_data()
        train_x, train_y = trainer.create_training_data()
        trainer.build_model(train_x, train_y)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()