# RAG - Système de recommandation de films TMDB

Un Agent de recommandation de films basé sur le dataset TMDB 5000.

---

## Comment lancer le projet

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

`requirements.txt` :
```
sentence-transformers
torch
numpy
chromadb
groq
python-dotenv
pandas
```

### 2. Clé API Groq

Créer un fichier `.env` à la racine du projet :

```
GROQ_API_KEY=ta_clé_ici
```

### 3. Indexation (à faire une seule fois)

```bash
python indexation.py
```

Cela charge le CSV, construit les textes, les embedde et crée la base vectorielle dans `./tmdb_vector_db/`. Si la base existe déjà, l'indexation est ignorée automatiquement.

### 4. Lancer l'assistant

```bash
python rag.py
```


Une fois lancé, le système affiche une invite de commande interactive. Tu peux poser autant de questions que tu veux sur les films. Pour quitter, tape quit, exit ou q.

```
Assistant films TMDB prêt. Tapez 'quit' pour quitter.

Votre question : Je cherche un thriller psychologique
...
Votre question : quit
```

## L'embedding

Le modèle utilisé est `distiluse-base-multilingual-cased-v2`


---


## RQ : Pas de chunking

Le chunking consiste à découper un document long en morceaux cohérents avant de les embedder. C'est utile quand les documents sont longs (pdf, articles, etc.)

Dans notre cas, on a pour chaque film des court texte à assembler ensemble (titre, genres, overview, etc.) — le résultat final fait environ 300-500 tokens, ce qui est largement gérable par les LLM.

---

## Le système RAG

### Format du contexte injecté au LLM

Pour chaque question, on embedde la question avec le même modèle, on cherche les 3 films les plus proches dans ChromaDB, et on injecte leurs textes dans le prompt système :

```
Tu es un assistant expert en recommandation de films.

Voici les films les plus pertinents :

--- Film 1 ---
Titre: Shutter Island | Année: 2010 | Genres: Thriller, Mystery | Rating: 8.1/10 | ...

--- Film 2 ---
Titre: Black Swan | Année: 2010 | Genres: Thriller, Drama | Rating: 7.2/10 | ...

--- Film 3 ---
Titre: Gone Girl | Année: 2014 | Genres: Thriller, Mystery | Rating: 8.1/10 | ...
```

Le format `--- Film N ---` a été choisi pour que le LLM distingue clairement les différents films dans son contexte.


---

## Structure du projet (pushé sur github)

```

├── indexation.py        # Chargement CSV, embedding, création base vectorielle
├── rag.py               # Agent RAG : recherche + génération de réponses
├── context.txt          # Prompt système du LLM
├── tmdb_5000_movies.csv # Dataset TMDB
├── requirements.txt     # Dépendances Python
└── README.md
```

---

#### Q1. Comment convertir chaque ligne CSV en texte embedable ?

Les données TMDB sont tabulaires — chaque film est une ligne avec des colonnes séparées. Le modèle d'embedding ne comprend que du texte continu, il a donc fallu "aplatir" chaque ligne en une phrase descriptive cohérente.

**Colonnes incluses dans le texte embedé :**

| Colonne        | Raison                                                 |
| -------------- | ------------------------------------------------------ |
| `title`        | Identifiant principal du film                          |
| `release_date` | Permet les recherches par période ("sorti après 2010") |
| `genres`       | Critère de recherche majeur ("thriller", "animation")  |
| `keywords`     | Capture l'ambiance et les thèmes précis du film        |
| `vote_average` | Permet de filtrer par qualité ("bien noté")            |
| `runtime`      | Utile pour certaines requêtes                          |
| `tagline`      | Résume l'esprit du film en une phrase                  |
| `overview`     | Description principale — le plus riche sémantiquement  |

**Colonnes ignorées dans l'embedding (gardées uniquement en métadonnées) :**

`budget`, `revenue`, `production_companies`, `production_countries`, `spoken_languages`, `homepage` — ces informations n'aident pas à la recherche sémantique.

**Format du texte construit pour chaque film :**

```
Titre: Avatar | Année: 2009 | Genres: Action, Adventure, Fantasy, Science Fiction |
Mot-clés: culture clash, space war, alien... | Rating: 7.2/10 | Durée: 162 min |
Tagline: Enter the World of Pandora. | Overview: In the 22nd century, a paraplegic Marine...
```

#### Q2. Comment parser les genres et keywords au format JSON imbriqué ?

Les colonnes `genres` et `keywords` sont des json imbriqué dans le CSV :

```
"[{""id"": 28, ""name"": ""Action""}, {""id"": 12, ""name"": ""Adventure""}]"
```

Pandas lit ça comme une simple string c'est pour ça qu'il faut d'abord la parser avec `json.loads()` pour obtenir une vraie liste Python, puis extraire uniquement les champs `"name"` :

```python
def parse_json_names(json_str):
    try:
        items = json.loads(json_str)
        return ", ".join([item["name"] for item in items])
    except:
        return ""

df["genres_parsed"] = df["genres"].apply(parse_json_names)
df["keywords_parsed"] = df["keywords"].apply(parse_json_names)
```

Résultat : `"Action, Adventure, Fantasy, Science Fiction"` — une string propre directement intégrable dans le texte embedé.

---

#### Q3. Stratégie de persistance

L'indexation de 4803 films prend plusieurs minutes. Pour ne pas avoir à la relancer à chaque test, deux mécanismes sont en place :

**ChromaDB PersistentClient** — la base est sauvegardée sur disque dans `./tmdb_vector_db/` et rechargée automatiquement à chaque lancement de `rag.py`. Aucun recalcul des vecteurs n'est nécessaire.

**Vérification avant indexation** — dans `indexation.py`, avant d'appeler `collection.add()`, on vérifie si la collection est déjà remplie :

```python
if collection.count() > 0:
    print(f"Base déjà existante avec {collection.count()} films. Pas de réindexation.")
    return
```

et ca dans la fonction **store_in_chromadb**

---


#### Q4. Comment guider le LLM pour des recommandations pertinentes ?

Dans le prompt système (`context.txt`) on a indiqué :

- De répondre **que** sur la base des films fournis et aucune invention
- De **Citer** le titre, l'année, les genres et la note pour chaque recommandation
- D'**Expliquer** pourquoi le film correspond à la demande
- De **Corriger** poliment l'utilisateur si une information est erronée

La première version du prompt était trop permissive (celle utilsé pedant le cours) — le LLM recommandait des films même pour des questions hors sujet (ex: "meilleur restaurant à Paris" → il recommandait un film sur la cuisine). On a ajouté une règle explicite : si la question ne concerne pas les films, refuser clairement.

#### Q5. Que faire si l'utilisateur demande un film de 2024 ?

Le dataset TMDB utilisé couvre les films jusqu'en 2017. Si un utilisateur demande un film très récent, le système ne le trouvera pas dans sa base. Le comportement attendu est que le LLM réponde honnêtement qu'il ne dispose pas de cette information, conformément à la règle "ne répondre que sur la base des films fournis". Il ne doit pas inventer ni halluciner un film qu'il ne connaît pas.




---

## Exemples de questions

```
Je cherche un thriller psychologique avec un retournement de situation inattendu
Un film d'animation familial sorti après 2010
Un film comme Inception mais plus accessible ?
Je cherche Avatar, c'est un film sorti en 2015 non ?  ← le système corrige
Quel est le meilleur restaurant à Paris ?             ← le système refuse
```