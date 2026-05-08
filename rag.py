from groq import Groq
from dotenv import load_dotenv
import os
from indexation import VectorDB
from config import LLM_MODEL_NAME, DB_PATH, CSV_PATH


class RAG:
    def __init__(self, vector_db_name):
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.vector_db = VectorDB(vector_db_name, csv_path=CSV_PATH)

    @staticmethod
    def read_file(file_path):
        with open(file_path, "r") as file:
            return file.read()

    def build_context(self, question):
        chunks, metadatas = self.vector_db.retrieve(question, n=5)

        chunks_formates = ""
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
            chunks_formates += f"\n--- Film {i+1} ---\n{chunk}\n"

        context = RAG.read_file("context.txt")
        return context.replace("{{Chuncks}}", chunks_formates)

    def answer_question(self, question):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.build_context(question),
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model=LLM_MODEL_NAME
        )
        return chat_completion.choices[0].message.content


if __name__ == "__main__":
    rag = RAG(vector_db_name=DB_PATH)
    print("Assistant films TMDB prêt. Tapez 'quit' pour quitter.\n")
    while True:
        question = input("Votre question : ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue
        print("\n" + rag.answer_question(question) + "\n")