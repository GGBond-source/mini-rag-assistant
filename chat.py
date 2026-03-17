import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from retriever import Retriever
from rag_pipeline import answer_question


def print_banner():

    print("\n==============================")
    print(" Mini RAG Assistant")
    print(" type 'exit' to quit")
    print("==============================\n")


def print_answer(result):

    gen = result["generation_result"]

    print("\nAI:", gen["answer"])

    if gen["evidence"]:

        print("\nEvidence:")

        for i, e in enumerate(gen["evidence"], 1):
            print(f"{i}. {e}")

    print("\n------------------------------\n")


def main():

    retriever = Retriever()

    print_banner()

    while True:

        question = input("User: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nBye 👋")
            break

        if question == "":
            continue

        result = answer_question(question, retriever)

        print_answer(result)


if __name__ == "__main__":
    main()