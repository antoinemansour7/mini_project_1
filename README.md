# Student Support Assistant

This project is a small command-line assistant for student support questions.
It loads a CSV knowledge base, finds the closest matching answer with sentence embeddings, and checks the user's sentiment.
Strong negative sentiment triggers a recommendation to contact a human advisor.

## Setup

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run

Run it from this folder with:

```bash
python assistant.py
```

The first run may take longer because the embedding and sentiment models are downloaded and cached locally.
