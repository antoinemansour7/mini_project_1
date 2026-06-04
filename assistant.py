"""
COMP 472 - Mini Project 1
Intelligent Student Support Assistant
with Sentiment Analysis and Semantic Retrieval
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


def load_knowledge_base(filepath: str) -> tuple[list[str], list[str]]:
    """Load questions and answers from a CSV file."""
    try:
        df = pd.read_csv(filepath)

        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV must have 'question' and 'answer' columns.")

        questions = df["question"].tolist()
        answers   = df["answer"].tolist()

        print(f"[✓] Knowledge base loaded: {len(questions)} entries from '{filepath}'")
        return questions, answers

    except FileNotFoundError:
        print(f"[✗] Error: File '{filepath}' not found.")
        raise
    except Exception as e:
        print(f"[✗] Error loading knowledge base: {e}")
        raise


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the embedding model used for semantic search."""
    print(f"[✓] Loading embedding model: '{model_name}' ...")
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(texts, show_progress_bar=False)
    print(f"[✓] Generated embeddings: matrix shape {embeddings.shape}")
    return embeddings


def find_best_answer(
    user_query: str,
    model: SentenceTransformer,
    question_embeddings: np.ndarray,
    answers: list[str],
    threshold: float = 0.3,
) -> str:
    """Retrieve the most semantically relevant answer for a user query."""
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, question_embeddings)

    best_index = int(np.argmax(similarities[0]))
    best_score = float(similarities[0][best_index])

    if best_score < threshold:
        return "I'm sorry, I couldn't find a relevant answer. Please contact support directly."

    return answers[best_index]


def load_sentiment_pipeline() -> pipeline:
    """Load the sentiment analysis pipeline."""
    print("[✓] Loading sentiment analysis model ...")
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline


def analyze_sentiment(sentiment_pipeline, text: str) -> tuple[str, float]:
    result = sentiment_pipeline(text)[0]
    label  = result["label"]
    score  = round(result["score"], 4)
    return label, score


def should_escalate(label: str, score: float, threshold: float = 0.9) -> bool:
    return label == "NEGATIVE" and score >= threshold


def update_history(history: list[dict], user_input: str, bot_answer: str) -> None:
    history.append({"user": user_input, "bot": bot_answer})


def print_history(history: list[dict]) -> None:
    """Print a summary of the conversation so far."""
    print("\n" + "═" * 50)
    print("  CONVERSATION HISTORY")
    print("═" * 50)
    if not history:
        print("  (no messages yet)")
    for i, turn in enumerate(history, 1):
        print(f"  [{i}] You : {turn['user']}")
        print(f"       Bot : {turn['bot']}")
        print()


def main():
    KNOWLEDGE_BASE_PATH = "knowledge_base.csv"

    print("\n" + "═" * 50)
    print("  Initializing Student Support Assistant...")
    print("═" * 50)

    questions, answers         = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    embedding_model            = load_embedding_model("all-MiniLM-L6-v2")
    question_embeddings        = generate_embeddings(embedding_model, questions)
    sentiment_model            = load_sentiment_pipeline()

    conversation_history: list[dict] = []

    print("\n" + "═" * 50)
    print("  Welcome to Concordia Student Support AI  ")
    print("  Type 'quit' to exit | 'history' to review")
    print("═" * 50 + "\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("\n[✓] Thank you for using Student Support AI. Goodbye!\n")
            break

        if user_input.lower() == "history":
            print_history(conversation_history)
            continue

        if not user_input:
            print("Please type a question.\n")
            continue

        label, score = analyze_sentiment(sentiment_model, user_input)
        print(f"Sentiment: {label} ({score:.2f})")

        if should_escalate(label, score):
            print("⚠  Recommended escalation: Please contact a human advisor.")
            print("   📞 Concordia Ombuds Office: 514-848-2424 ext. 3786\n")

        answer = find_best_answer(
            user_input,
            embedding_model,
            question_embeddings,
            answers,
        )
        print(f"Answer: {answer}\n")

        update_history(conversation_history, user_input, answer)


if __name__ == "__main__":
    main()
