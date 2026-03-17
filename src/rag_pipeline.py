from retriever import Retriever
from prompt_builder import build_prompt
from generator import generate_answer


def answer_question(question, retriever, top_k=5, top_n=3):
    recalled_docs = retriever.search_documents(question, top_k=top_k)
    reranked_docs = retriever.rerank_documents(question, recalled_docs)

    contexts = [doc["text"] for doc in reranked_docs[:top_n]]
    prompt = build_prompt(question, contexts)
    generation_result = generate_answer(question, prompt, contexts)

    return {
        "question": question,
        "recalled_docs": recalled_docs,
        "reranked_docs": reranked_docs,
        "contexts": contexts,
        "prompt": prompt,
        "generation_result": generation_result,
    }


def pretty_print_result(result):
    print("=" * 70)
    print("Question:")
    print(result["question"])

    print("\n[Stage1: Recall]")
    for i, doc in enumerate(result["recalled_docs"], 1):
        print(
            f"{i}. id={doc['id']} | score={doc['score']:.4f} | text={doc['text']}"
        )

    print("\n[Stage2: Rerank]")
    for i, doc in enumerate(result["reranked_docs"], 1):
        print(
            f"{i}. id={doc['id']} | rerank_score={doc['rerank_score']:.4f} | text={doc['text']}"
        )

    print("\n[Contexts for Generation]")
    for i, c in enumerate(result["contexts"], 1):
        print(f"{i}. {c}")

    gen = result["generation_result"]

    print("\n[Generation Result]")
    print(f"status: {gen['status']}")
    print(f"answer: {gen['answer']}")
    print("evidence:")
    if gen["evidence"]:
        for e in gen["evidence"]:
            print(f"- {e}")
    else:
        print("- None")

    print("=" * 70)


def main():
    retriever = Retriever()

    question = input("请输入问题：").strip()
    if not question:
        question = "猫是什么动物？"

    result = answer_question(question, retriever)
    pretty_print_result(result)


if __name__ == "__main__":
    main()