# RAG — Système de recommandation de films TMDB

## 1. Problématique et solution

Un LLM classique comme Llama est entraîné sur des données jusqu'à une certaine date. Il ne connaît pas vos documents internes, ne peut pas répondre sur une base de données spécifique, et invente des réponses quand il ne sait pas — c'est ce qu'on appelle une **hallucination**.

Ce projet implémente un **RAG (Retrieval-Augmented Generation)** — une technique qui enrichit le contexte du LLM avec des informations récupérées dynamiquement depuis une base de connaissances. Au lieu de répondre de mémoire, le LLM consulte les bons documents avant de répondre.

**Ce que ça résout concrètement** : un utilisateur peut poser des questions en langage naturel sur des films et obtenir des recommandations basées sur une vraie base de données, sans hallucination.

Exemples de requêtes supportées :
- *"Je cherche un thriller psychologique avec un retournement de situation inattendu"*
- *"Recommande-moi un film d'animation familial sorti après 2010"*
- *"Un film comme Inception mais plus accessible ?"*

---

## 2. Dataset

Le projet utilise le [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) disponible sur Kaggle, composé de deux fichiers :

**`tmdb_5000_movies.csv`** — 4803 films avec leurs métadonnées (titre, synopsis, genres, note, durée, etc.)

**`tmdb_5000_credits.csv`** — cast et équipe technique de chaque film. Ce fichier n'a pas été intégré au pipeline pour deux raisons : la colonne `cast` contient des dizaines d'acteurs par film, difficiles à indexer sans polluer l'embedding ; la colonne `crew` contient le réalisateur mais sa valeur ajoutée était limitée par rapport à la complexité du merge. C'est une piste d'amélioration identifiée.

### Colonnes incluses dans le texte embedé

| Colonne                       | Raison                                      |
| ----------------------------- | ------------------------------------------- |
| `title` / `original_title`    | Identification du film                      |
| `genres`                      | Critère de recherche principal (JSON parsé) |
| `keywords`                    | Thèmes et ambiance du film (JSON parsé)     |
| `overview`                    | Synopsis — la plus riche sémantiquement     |
| `tagline`                     | Accroche — capture l'esprit du film         |
| `vote_average` / `vote_count` | Note et popularité                          |
| `release_date`                | Année de sortie                             |
| `runtime`                     | Durée                                       |
| `original_language`           | Langue originale                            |

### Colonnes ignorées

`budget`, `revenue`, `production_companies`, `production_countries`, `spoken_languages`, `homepage` — pas utiles pour la recherche sémantique.

---

## 3. Structure du repo

```
.
├── indexation.py       # Classe VectorDB — chargement CSV, embedding, base vectorielle
├── rag.py              # Classe RAG — recherche + génération de réponses
├── config.py           # Constantes centralisées (modèles, chemins)
├── context.txt         # Prompt système du LLM
├── requirements.txt    # Dépendances Python
├── README.md
└── compte_rendu.pdf
```

---

## 4. Comment lancer le projet

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

Créer un fichier `.env` à la racine :

```
GROQ_API_KEY=ta_clé_ici
```


### 3. Lancer l'assistant

```bash
python rag.py
```

Au premier lancement, la base vectorielle est créée automatiquement depuis le CSV — cela prend quelques minutes. Aux lancements suivants, elle est rechargée instantanément depuis le disque.

Une fois lancé, le système affiche une invite interactive. Pour quitter, tapez `quit`, `exit` ou `q`.

```
Assistant films TMDB prêt. Tapez 'quit' pour quitter.

Votre question : Je cherche un thriller psychologique
...
Votre question : quit
```

---

## 5. Modèles utilisés

### Modèle d'embedding

**Modèle actuel : `paraphrase-multilingual-mpnet-base-v2`**

Après avoir utilisé `distiluse-base-multilingual-cased-v2` (512 dimensions) (un modèle léger et rapide) et après, c'est `paraphrase-multilingual-mpnet-base-v2`  qui a été utilisé(768 dimensions) pour deux raisons :

- **Plus performant** : 768 dimensions vs 512 — vecteurs plus riches, meilleure capture des nuances sémantiques
- **Mieux optimisé pour le multilingue** : les requêtes en français retrouvent mieux les films dont les données sont en anglais

La migration nécessite de supprimer la base vectorielle et de réindexer depuis zéro.

**Note importante** : le nom du modèle est sauvegardé dans les métadonnées de la collection ChromaDB. Au rechargement, le système relit automatiquement le bon modèle — impossible d'utiliser accidentellement un modèle différent de celui utilisé à l'indexation.

### Modèle LLM

**`llama-3.3-70b-versatile`** via l'API Groq. Choisi pour sa qualité de formulation et sa capacité à suivre des instructions complexes dans le prompt système. Le tier gratuit Groq offre une vitesse d'inférence élevée sans frais.

---

## 6. Détails des fichiers

### `config.py`
Centralise toutes les constantes du projet — nom du modèle d'embedding, modèle LLM, chemin de la base, nom de la collection, chemin du CSV. Principe DRY : une seule valeur à changer pour impacter tout le projet.

### `indexation.py` — classe `VectorDB`
Gère toute la logique de la base vectorielle :
- **`__init__`** : si la base existe sur disque → `load_vector_db()`, sinon → `create_vector_db()`. C'est le mécanisme d'idempotence — on ne réindexe jamais inutilement.
- **`create_vector_db`** : charge le CSV, parse les colonnes JSON, construit les textes, embedde, stocke dans ChromaDB.
- **`load_vector_db`** : recharge la base existante et relit le modèle d'embedding depuis les métadonnées.
- **`retrieve`** : embedde la question et retourne les n films les plus proches par similarité cosinus.

### `rag.py` — classe `RAG`
Gère le système de questions-réponses :
- **`__init__`** : initialise le client Groq et instancie `VectorDB`.
- **`build_context`** : appelle `retrieve()`, formate les films avec `--- Film N ---` et injecte dans le prompt système.
- **`answer_question`** : envoie le contexte + la question à l'API Groq et retourne la réponse.

### `context.txt`
Prompt système du LLM. Définit le comportement de l'assistant : ne répondre que sur la base des films fournis, citer titre/année/genres/note, expliquer pourquoi le film correspond, refuser les questions hors sujet, corriger poliment les informations erronées.

---

## 7. Réponses aux questions du sujet

#### Q1. Comment convertir chaque ligne CSV en texte embedable ?

Les données TMDB sont tabulaires — chaque film est une ligne avec des colonnes séparées. Le modèle d'embedding ne comprend que du texte continu, il a donc fallu "aplatir" chaque ligne en une phrase descriptive cohérente en assemblant les colonnes les plus utiles sémantiquement (voir section 2).

#### Q2. Comment parser les genres et keywords au format JSON imbriqué ?

Les colonnes `genres` et `keywords` contiennent du JSON imbriqué dans le CSV. Pandas les lit comme des strings brutes. On applique `json.loads()` pour convertir en liste Python, puis on extrait les champs `"name"` :

```python
def parse_json_names(json_str):
    try:
        items = json.loads(json_str)
        return ", ".join([item["name"] for item in items])
    except:
        return ""
```

#### Q3. Stratégie de persistance

Deux mécanismes évitent de réindexer à chaque lancement :

**`os.path.exists()`** dans le constructeur `VectorDB.__init__` — si le dossier de la base existe, on la recharge sans recalculer les embeddings.

**Modèle sauvegardé dans les métadonnées ChromaDB** — au rechargement, le nom du modèle est relu depuis la collection pour garantir la cohérence.

#### Q4. Comment guider le LLM pour des recommandations pertinentes ?

Le prompt système (`context.txt`) impose des règles strictes : ne répondre que sur les films fournis, citer titre/année/genres/note, expliquer le choix, refuser les questions hors sujet, corriger les erreurs de l'utilisateur. La première version était trop permissive — le LLM recommandait des films même pour des questions hors sujet.

#### Q5. Que faire si l'utilisateur demande un film de 2024 ?

Le dataset couvre les films jusqu'en 2017. Le LLM doit répondre honnêtement qu'il ne dispose pas de cette information — règle explicite dans le prompt système.

---

## 8. Types de questions et limites

### Questions qui fonctionnent bien 

```
Je cherche un thriller psychologique avec un retournement de situation inattendu
Un film d'animation familial
Un film comme Inception mais plus accessible ?
Un film avec une ambiance sombre et oppressante
Un film de science-fiction sur l'intelligence artificielle
Un film romantique avec une happy end
Spiderman
```

Ces questions fonctionnent car elles décrivent une **ambiance, un thème ou un genre** — le modèle d'embedding excelle à la similarité sémantique sur ce type de requêtes.

### Questions problématiques

```
Classe les films les mieux notés
Un film sorti après 2010
Un film avec Leonardo DiCaprio
Un film de Christopher Nolan
Le film le plus long
Les films avec plus de 10 000 votes
```

**Pourquoi ça ne fonctionne pas** : le modèle d'embedding ne comprend pas les relations numériques ("supérieur à", "meilleur que") ni les contraintes temporelles strictes. Il cherche par similarité sémantique — pas par filtres. La solution serait d'ajouter des filtres sur les métadonnées ChromaDB (`where={"note": {"$gte": 8.0}}`), mais cela nécessite de détecter automatiquement l'intention de la question (query parsing) — une amélioration hors scope pour ce projet.

De plus, le réalisateur et les acteurs ne sont pas indexés (voir section 2), ce qui explique les mauvais résultats sur ces critères.