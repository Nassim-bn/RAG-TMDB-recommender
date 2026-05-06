from groq import Groq
from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialisé juste 1 seule fois au chargement du script
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
chroma = chromadb.PersistentClient(path="./tmdb_vector_db")
collection = chroma.get_or_create_collection("films")


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def retrieve(question, n=5):
    # On embedde la question avec le MÊME modèle qu'à l'indexation
    # On met en minuscules car les textes indexés sont aussi en minuscules
    embedded_question = model.encode(
        [question.lower()],
        normalize_embeddings=True
    ).tolist()[0]

    results = collection.query(query_embeddings=[embedded_question], n_results=n)

    # results["documents"] est [[film1, film2, film3]]
    # le [0] récupère la liste intérieure
    return results["documents"][0], results["metadatas"][0]


def build_context(question):
    chunks, metadatas = retrieve(question, n=5)

    chunks_formates = ""
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        chunks_formates += f"\n--- Film {i+1} ---\n{chunk}\n"

    context = read_file("context.txt")
    return context.replace("{{Chuncks}}", chunks_formates)


def answer_question(question):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": build_context(question),
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    print("Assistant films TMDB prêt. Tapez 'quit' pour quitter.\n")
    while True:
        question = input("Votre question : ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue
        print("\n" + answer_question(question) + "\n")