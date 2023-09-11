# Question-Answering Language Model (QA-LLM) Chatbot

[[Chatbot Demo]](https://github.com/rajan1994/QA-LLM-Chatbot/tree/main/Project%20-%20QA%20on%20Private%20Documents/demo/Project%20Demo.webm)


## Introduction

This repository contains a Question-Answering Language Model (QA-LLM) Chatbot powered by GPT-3.5, a state-of-the-art natural language processing model developed by OpenAI. This chatbot is designed to answer questions and engage in natural language conversations.

## Features

- **Interactive Chat**: Engage in conversations with the chatbot by typing questions or statements.
- **Dynamic Responses**: The chatbot provides context-aware responses based on the input.
- **Easy Integration**: Easily integrate this chatbot into your projects and applications.

## Getting Started

### Prerequisites

- Python 3.x
- OpenAI GPT-3.5 API key (Sign up at https://beta.openai.com/ to get your API key)

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/rajan1994/QA-LLM-Chatbot.git
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key as an environment variable:

   ```bash
   export OPENAI_API_KEY=your-api-key
   export PINECONE_API_KEY=your-pinecone-api-key
   export PINECONE_ENV=your-pinecone-env Ex:"asia-southeast1-gcp-free"
   ```

### Usage

1. Run the chatbot script:

   ```bash
   Run code in `QA-LLM-Chatbot/Project - QA on Private Documents/question_answering_on_private_data.ipynb`
   ```

2. Start interacting with the chatbot by typing your questions or statements.

### Configuration

You can customize the chatbot's behavior by modifying the `config.py` file. Adjust parameters like conversation history length, temperature, and more to fine-tune the responses.

## Contributing

We welcome contributions! If you want to improve this chatbot, please follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make your changes and commit them: `git commit -m 'Add your feature'`.
4. Push to your forked repository: `git push origin feature/your-feature-name`.
5. Create a pull request, describing your changes in detail.


## Acknowledgments

- Thanks to OpenAI for providing the GPT-3.5 model.
- Inspired by the awesome possibilities of natural language processing and chatbots!

---

Feel free to customize this README to include additional information specific to your repository. Make sure to update the placeholders like `your-username` and `your-api-key` with your actual information and specifics about your project.