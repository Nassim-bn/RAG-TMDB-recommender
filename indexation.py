import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from config import EMBEDDING_MODEL_NAME, COLLECTION_NAME


class VectorDB:
    def __init__(self, vector_db_name, csv_path=None):
        if os.path.exists(vector_db_name):
            self.load_vector_db(vector_db_name)
        elif csv_path:
            self.create_vector_db(vector_db_name, csv_path)
        else:
            raise Exception("Donnez un chemin vers une base existante ou un CSV ")

    def load_vector_db(self, vector_db_name):
        print("Chargement de la base vectorielle existante...")
        self.chroma = chromadb.PersistentClient(path=vector_db_name)
        collection_info = self.chroma.get_collection(COLLECTION_NAME)
        model_name = collection_info.metadata["embedding_model"]
        print(f"Modèle d'embedding : {model_name}")
        self.model = SentenceTransformer(model_name)

    def create_vector_db(self, vector_db_name, csv_path):
        print("Création de la base vectorielle...")
        print(f"Modèle d'embedding : {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chroma = chromadb.PersistentClient(path=vector_db_name)

        # On sauvegarde le nom du modèle dans les métadonnées de la collection
        # Comme ça au rechargement on est certain d'utiliser le même modèle
        collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": EMBEDDING_MODEL_NAME}
        )

        df = self.load_and_prepare_dataframe(csv_path)
        documents = self.build_documents(df)
        ids, chunks, metadatas = self.decompress_documents(documents)
        embeddings = self.get_embeddings(chunks, shw_progress_bar=True)

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Base vectorielle créée avec {collection.count()} films indexés !")

    def load_and_prepare_dataframe(self, csv_path):
        """
        Charge le CSV et parse les colonnes JSON (genres, keywords)
        """
        df = pd.read_csv(csv_path)

        def parse_json_names(json_str):
            try:
                items = json.loads(json_str)
                return ", ".join([item["name"] for item in items])
            except:
                return ""

        df["genres_parsed"] = df["genres"].apply(parse_json_names)
        df["keywords_parsed"] = df["keywords"].apply(parse_json_names)

        print(f"Dataset chargé : {len(df)} films trouvés.")
        return df

    def build_documents(self, df):
        """
        Construit la liste de dictionnaires {id, contenu, metadata} à partir du DataFrame.
        """
        documents = []
        for index, row in df.iterrows():
            annee = str(row["release_date"]).split("-")[0] if pd.notna(row["release_date"]) else "Inconnu"

            texte = (
                f"Titre: {row['title']} | Titre original: {row['original_title']} | "
                f"Année: {annee} | Langue: {row['original_language']} | "
                f"Genres: {row['genres_parsed']} | Mots-clés: {row['keywords_parsed']} | "
                f"Note: {row['vote_average']}/10 | Votes: {row['vote_count']} | "
                f"Durée: {row['runtime']} min | Tagline: {row['tagline']} | "
                f"Synopsis: {row['overview']}"
            ).lower()

            documents.append({
                "id": f"movie_{row['id']}",
                "contenu": texte,
                "metadata": {
                    "source": "tmdb_5000_movies.csv",
                    "collection": COLLECTION_NAME,
                    "titre": str(row["title"]),
                    "annee": annee,
                    "genres": str(row["genres_parsed"]),
                    "note": float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0
                }
            })

        print(f"Documents construits : {len(documents)} films préparés.")
        return documents

    def decompress_documents(self, documents):
        """
        Décompresse la liste de dicts en 3 listes parallèles pour ChromaDB.
        """
        ids       = [doc["id"]       for doc in documents]
        chunks    = [doc["contenu"]  for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        return ids, chunks, metadatas

    def get_embeddings(self, chunks,shw_progress_bar=False):
        """
        Calcule les embeddings pour une liste de textes.
        """
        return self.model.encode(
            chunks,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=shw_progress_bar
        ).tolist()

    def retrieve(self, question, n=5):
        """
        Recherche les n chunks les plus proches de la question.
        """
        embedded_question = self.get_embeddings([question.lower()])[0]
        collection = self.chroma.get_collection(COLLECTION_NAME)
        results = collection.query(query_embeddings=[embedded_question], n_results=n)
        return results["documents"][0], results["metadatas"][0]


if __name__ == "__main__":
    vector_db = VectorDB(vector_db_name="./tmdb_vector_db", csv_path="tmdb_5000_movies.csv")