import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import chromadb


def load_and_prepare_dataframe(csv_path):
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


def build_documents(df, collection_name):
    """
    Construit la liste de dictionnaires {id, contenu, metadata} à partir du DataFrame.
    collection_name est sauvegardé dans les métadonnées pour tracer l'origine de chaque document.
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
                "collection": collection_name,
                "titre": str(row["title"]),
                "annee": annee,
                "genres": str(row["genres_parsed"]),
                "note": float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0
            }
        })

    print(f"Documents construits : {len(documents)} films préparés.")
    return documents


def get_embeddings(documents, model):
    """
    Décompresse les documents et calcule les embeddings.
    """
    ids       = [doc["id"]       for doc in documents]
    chunks    = [doc["contenu"]  for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    embeddings = model.encode(
        chunks,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    return ids, chunks, metadatas, embeddings


def store_in_chromadb(ids, chunks, metadatas, embeddings, db_path, collection_name):
    """
    Crée la base ChromaDB persistante et y stocke les vecteurs, textes et métadonnées.
    Idempotent : n'ajoute que les films non encore indexés.
    """
    chroma = chromadb.PersistentClient(path=db_path)
    collection = chroma.get_or_create_collection(collection_name)

    if collection.count() > 0:
        existing = collection.get(ids=ids)["ids"] #récupère tous les IDs déjà présents
        ids_to_add = [i for i in ids if i not in existing] #Liste vide car tous les films sont déjà là

        if not ids_to_add:
            print(f"Base déjà existante avec {collection.count()} films. Pas de réindexation.")
            return

        indices = [ids.index(i) for i in ids_to_add]
        collection.add(
            ids=ids_to_add,
            documents=[chunks[i] for i in indices],
            embeddings=[embeddings[i] for i in indices],
            metadatas=[metadatas[i] for i in indices]
        )
        print(f"{len(ids_to_add)} nouveaux films ajoutés.")
        return

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Base vectorielle créée avec {collection.count()} films indexés")


if __name__ == "__main__":
    COLLECTION_NAME = "films"
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    df = load_and_prepare_dataframe("tmdb_5000_movies.csv")
    documents = build_documents(df, COLLECTION_NAME)
    ids, chunks, metadatas, embeddings = get_embeddings(documents, model)
    store_in_chromadb(ids, chunks, metadatas, embeddings, "./tmdb_vector_db", COLLECTION_NAME)