import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import chromadb


def load_and_prepare_dataframe(csv_path):
    """
    Charge le CSV et parse les colonnes JSON (genres, keywords) avec les nouvelles colonnes genres_parsed et keywords_parsed
    """
    df = pd.read_csv(csv_path)

    def parse_json_names(json_str):
        try:
            items = json.loads(json_str)
            return ", ".join([item["name"] for item in items]) # On a le texte en string separé avec des ,
        except:
            return ""

    df["genres_parsed"] = df["genres"].apply(parse_json_names)
    df["keywords_parsed"] = df["keywords"].apply(parse_json_names)

    print(f"Dataset chargé : {len(df)} films trouvés.")
    return df


def build_documents(df):
    """
    Construit la liste de dictionnaires {id, contenu, metadata} à partir du DataFrame
    """
    documents = []
    for index, row in df.iterrows():
        # On garde juste l'année sans le jour et le mois
        annee = str(row["release_date"]).split("-")[0] if pd.notna(row["release_date"]) else "Inconnu"

        texte = (
            f"Titre: {row['title']} | Titre original: {row['original_title']} | "
            f"Année: {annee} | Langue: {row['original_language']} | "
            f"Genres: {row['genres_parsed']} | Mots-clés: {row['keywords_parsed']} | "
            f"Note: {row['vote_average']}/10 | Votes: {row['vote_count']} | "
            f"Durée: {row['runtime']} min | Tagline: {row['tagline']} | "
            f"Synopsis: {row['overview']}"
        ).lower()  # On met tout en minuscules pour bien retrouver les donnes

        documents.append({
            "id": f"movie_{row['id']}",
            "contenu": texte,
            "metadata": {
                "source": "tmdb_5000_movies.csv",
                "titre": str(row["title"]),
                "annee": annee,
                "genres": str(row["genres_parsed"]),
                "note": float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0
            }
        })

    print(f"Documents construits : {len(documents)} films préparés.")
    return documents


def get_embeddings(documents,model=SentenceTransformer):
    """

    Embedding avec le modèle SentenceTransformer.
    """
    #  décompression de la liste de dicts en trois listes séparées (c'est ce que ChromaDB attend)
    ids= [doc["id"] for doc in documents]
    chunks = [doc["contenu"]  for doc in documents]
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
    Si la collection existe déjà, on ne réindexe pas pour éviter les doublons.
    """
    chroma = chromadb.PersistentClient(path=db_path)
    collection = chroma.get_or_create_collection(collection_name)

    # Si la collection est déjà remplie, pas besoin de réindexer
    if collection.count() > 0:
        print(f"Base déjà existante avec {collection.count()} films. Pas de réindexation.")
        return

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Base vectorielle créée avec {collection.count()} films indexés")


if __name__ == "__main__":
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    df = load_and_prepare_dataframe("tmdb_5000_movies.csv")
    documents = build_documents(df)
    ids, chunks, metadatas, embeddings = get_embeddings(documents, model)
    store_in_chromadb(ids, chunks, metadatas, embeddings, "./tmdb_vector_db", "films")