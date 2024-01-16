# Sunbot-Xeven-Hackathon
Sun Bot is an AI Chat Bot that allows users to upload document and ask questions. This is developed for Xeven Hackathon Competition.

# SunBot Local Setup Guide

This guide provides instructions on how to set up and run the Sunbot locally on your machine.

## Prerequisites

Ensure you have the following prerequisites installed on your machine:

- Python (version 3.9)
- Pip (Python package installer)
- Git

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/SunnyRehman/sunbot-xeven-hackathon
    ```

2. Navigate to the project directory:

    ```bash
    cd your-chatbot-repo
    ```

3. Create an Anaconda Environment
    ```bash
    conda create -n name_of_environment python=3.9
    ```
5. Activate the Conda Environment

  ```bash
    conda activate name_of_environment
    ```

4. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. Set up environment variables:

    Create a `.streamlit` directory and create a `secrets.toml` file inside it and add the following:

    ```dotenv
    OPENAI_API_KEY=your_openai_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_ENV=your_pinecone_environment
    ```

    Replace `your_openai_api_key`, `your_pinecone_api_key`, and `your_pinecone_environment` with your actual API keys.

## Running the Chatbot

1. Execute the following command to start the chatbot:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to http://localhost:8501

3. Interact with the chatbot by following the on-screen instructions
