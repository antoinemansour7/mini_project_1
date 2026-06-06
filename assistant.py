"""
COMP 472 - Mini Project 1
Intelligent Student Support Assistant
with Sentiment Analysis and Semantic Retrieval
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = PROJECT_DIR / "data" / "knowledge_base.csv"


def load_knowledge_base(filepath: str | Path) -> tuple[list[str], list[str]]:
    """Load questions and answers from a CSV file."""
    try:
        df = pd.read_csv(filepath)

        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV must have 'question' and 'answer' columns.")

        questions = df["question"].dropna().astype(str).tolist()
        answers = df["answer"].dropna().astype(str).tolist()

        if len(questions) != len(answers):
            raise ValueError("CSV question and answer columns must have the same number of values.")
        if not questions:
            raise ValueError("CSV knowledge base is empty.")

        print(f"[OK] Knowledge base loaded: {len(questions)} entries from '{filepath}'")
        return questions, answers

    except FileNotFoundError:
        print(f"[ERROR] File '{filepath}' not found.")
        raise
    except Exception as error:
        print(f"[ERROR] Error loading knowledge base: {error}")
        raise


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the embedding model used for semantic search."""
    print(f"[OK] Loading embedding model: '{model_name}' ...")
    return SentenceTransformer(model_name)


def generate_embeddings(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Generate embeddings for knowledge-base questions."""
    embeddings = model.encode(texts, show_progress_bar=False)
    print(f"[OK] Generated embeddings: matrix shape {embeddings.shape}")
    return embeddings


def _query_tokens(text: str) -> set[str]:
    """Keep meaningful words for overlap scoring."""
    return {word for word in text.lower().split() if len(word) > 2}


def find_best_answer(
    user_query: str,
    model: SentenceTransformer,
    questions: list[str],
    question_embeddings: np.ndarray,
    answers: list[str],
    threshold: float = 0.3,
) -> str:
    """Retrieve the most semantically relevant answer for a user query."""
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, question_embeddings)[0]

    query_words = _query_tokens(user_query)
    boosted_scores = similarities.copy()
    for i, question in enumerate(questions):
        overlap = len(query_words & _query_tokens(question))
        boosted_scores[i] += 0.04 * overlap

    best_index = int(np.argmax(boosted_scores))
    best_score = float(similarities[best_index])

    if best_score < threshold:
        return "I'm sorry, I couldn't find a relevant answer. Please contact support directly."

    return answers[best_index]


def load_sentiment_pipeline(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> pipeline:
    """Load the Hugging Face sentiment-analysis pipeline."""
    print(f"[OK] Loading sentiment analysis model: '{model_name}' ...")
    return pipeline("sentiment-analysis", model=model_name)


def _contains_keyword(text: str, keywords: frozenset[str]) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def analyze_sentiment(sentiment_pipeline, text: str) -> tuple[str, float]:
    """Return POSITIVE, NEUTRAL, or NEGATIVE with confidence score."""
    result = sentiment_pipeline(text, truncation=True)[0]
    model_label = result["label"].upper()
    model_score = round(result["score"], 4)

    if _contains_keyword(text, NEGATIVE_KEYWORDS):
        return "NEGATIVE", model_score if model_label == "NEGATIVE" else max(model_score, 0.9)

    if _contains_keyword(text, POSITIVE_KEYWORDS):
        return "POSITIVE", model_score if model_label == "POSITIVE" else max(model_score, 0.85)

    return "NEUTRAL", NEUTRAL_SENTIMENT_SCORE


def should_escalate(label: str, score: float, threshold: float = 0.9) -> bool:
    """Escalate strongly negative messages to a human advisor."""
    return label == "NEGATIVE" and score >= threshold


def update_history(history: list[dict[str, str]], user_input: str, bot_answer: str) -> None:
    """Add one conversation turn to the history."""
    history.append({"user": user_input, "bot": bot_answer})


def print_history(history: list[dict[str, str]]) -> None:
    """Print a summary of the conversation so far."""
    print("\n" + "=" * 50)
    print("  CONVERSATION HISTORY")
    print("=" * 50)
    if not history:
        print("  (no messages yet)")
    for index, turn in enumerate(history, 1):
        print(f"  [{index}] You : {turn['user']}")
        print(f"       Bot : {turn['bot']}")
        print()


def main() -> None:
    print("\n" + "=" * 50)
    print("  Initializing Student Support Assistant...")
    print("=" * 50)

    questions, answers = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    embedding_model = load_embedding_model("all-MiniLM-L6-v2")
    question_embeddings = generate_embeddings(embedding_model, questions)
    sentiment_model = load_sentiment_pipeline()

    conversation_history: list[dict[str, str]] = []

    print("\n" + "=" * 50)
    print("  Welcome to Concordia Student Support AI")
    print("  Type 'quit' to exit | 'history' to review")
    print("=" * 50 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("\n[OK] Thank you for using Student Support AI. Goodbye!\n")
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
            print("Recommended escalation: Please contact a human advisor.")
            print("Concordia Ombuds Office: 514-848-2424 ext. 3786\n")

        answer = find_best_answer(
            user_input,
            embedding_model,
            questions,
            question_embeddings,
            answers,
        )
        print(f"Answer: {answer}\n")

        update_history(conversation_history, user_input, answer)


NEGATIVE_KEYWORDS = frozenset({
    "frustrated", "frustrating", "frustration", "terrible", "angry", "anger",
    "hate", "hated", "awful", "horrible", "worst", "furious", "upset",
    "annoyed", "annoying", "ridiculous", "unacceptable", "disgusted",
    "unhappy", "disappointed", "disappointing", "infuriating", "outraged",
    "useless", "broken", "sucks", "stupid", "pathetic", "unfair", "not fair",
})
POSITIVE_KEYWORDS = frozenset({
    "thank", "thanks", "grateful", "great", "excellent", "wonderful",
    "happy", "love", "appreciate", "awesome", "perfect", "amazing",
    "fantastic", "pleased", "delighted", "helpful",
})
NEUTRAL_SENTIMENT_SCORE = 0.95


if __name__ == "__main__":
    main()
