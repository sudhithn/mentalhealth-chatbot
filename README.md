# DIY-NLP-Transformer-TensorFlow

This is the GPT (Basic version) branch.

# Generative Pre-trained Transformer (GPT)

The GPT (Generative Pre-trained Transformer) is a transformer-based deep learning model designed for natural language processing tasks. This implementation focuses on building a mental health chatbot that leverages GPT architecture principles for understanding user inputs and generating appropriate therapeutic responses. The model uses intent classification and natural language understanding to provide supportive conversations and mental health resources. It employs techniques like tokenization, word embeddings, and neural networks to process and respond to user queries effectively.

# Mental Health Chatbot

## Problem Definition

Develop a microservices-based architecture for a mental health chatbot using TensorFlow and NLTK to provide supportive conversations and mental health resources. The system uses intent classification to understand user inputs and generates appropriate responses based on a pre-trained model with therapeutic dialogue patterns.

## Data Definition

The intents.json dataset contains:
- Patterns: Various ways users might express their thoughts or feelings
- Responses: Appropriate therapeutic responses for each intent
- Tags: Categories of intents (e.g., greeting, depression, anxiety, stress)

Dataset Information:
- Source: Based on an open-source mental health chatbot intents dataset from GitHub
- Enhancement: Additional intents and responses added to cover more mental health scenarios
- Structure: JSON format organized by intent categories including general conversations, mental health topics, crisis support, and therapeutic responses

## Directory Structure

- **Code/bot.py**: Main script for model training and data preprocessing
- **Code/chat.py**: GUI implementation for user interaction
- **Data/intents.json**: Structured conversation data
- **Models/**: Directory containing saved model files and configurations

## Data Components

- **Training Data**: Intent patterns and responses
- **Model Files**: Saved neural network weights
- **Preprocessing Data**: Tokenized words and classes
- **Configuration Files**: Model parameters and settings

## Program Flow

1. **Data Preprocessing:** Process intent patterns, tokenize text, and create training data [`bot.py`]
2. **Model Training:** Train neural network on processed data using TensorFlow [`bot.py`]
3. **User Interface:** Provide GUI for user interactions and display responses [`chat.py`]
4. **Response Generation:** Generate appropriate responses based on intent classification
5. **Professional Resources:** Provide mental health resources and crisis support when needed

## Steps to Run

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies (choose one method):
```bash
# Method 1 - Using requirements.txt
pip install -r requirements.txt

# Method 2 - Install main required libraries
pip install tensorflow==2.18.0 nltk==3.9.1 numpy==2.0.2
```

3. Train the model:
```bash
python bot.py
```

4. Launch the chat interface:
```bash
python chat.py
```

## Features

- Real-time chat interface
- Intent-based response generation
- Mental health resource recommendations
- Crisis support detection and response
- Professional help referrals
- Multi-language support (basic)

## Model Architecture

- Input Layer: Word embeddings
- Hidden Layers: Dense layers with ReLU activation
- Dropout Layers: For regularization
- Output Layer: Softmax for intent classification