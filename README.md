
# Cahier de vacances Machine Learnia

Bienvenue dans le cahier de vacances Machine Learnia !

Chaque dimanche de l'été, un mini projet est publié. L'objectif est de pratiquer de façon concrète et progressive, les bases de l'analyse de données, le ML classique en passant par les RAG, et les graphes.

## Structure du repo

Chaque projet a son propre dossier, qui regroupe toutes ses ressources (notebook, données, images) :

```
cahier-de-vacances/
├── Projet_01/
│   └── projet_01.ipynb
├── Projet_02/
│   └── ...
├── pyproject.toml
└── README.md
```

## Faire le cahier de vacances depuis le début

### Étape 1 : cloner le projet

La première chose à faire est de récupérer le projet sur votre ordinateur. Ouvrez un terminal, placez-vous dans le dossier de votre choix et tapez :

```shell
git clone https://github.com/MachineLearnia/Cahier-Vacances-2026.git
```

Un dossier `Cahier-Vacances-2026` apparaît, c'est votre copie locale du projet. C'est aussi grâce à git que vous récupérerez les nouveaux projets chaque semaine (on en reparle plus bas).

### Étape 2 : installer uv

Pour l'environnement python, vous pouvez utiliser celui que vous préférez si vous êtes à l'aise avec, mais sinon, nous utiliserons le gestionnaire [`uv`](https://docs.astral.sh/uv/) qui est un outil récent et très populaire. Pour le télécharger, cela dépend de votre système d'exploitation. Si vous êtes sur Linux, MacOS ou WSL ouvrez votre terminal et tapez :

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Si vous êtes sur Windows ouvrez le powershell et tapez :

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Étape 3 : créer l'environnement

Le repo contient déjà tout ce qu'il faut pour construire l'environnement : le fichier `pyproject.toml` (qui décrit le projet et ses dépendances) et le fichier `.python-version` (qui fixe la version de python, ici la 3.13.2). Autrement dit, pas besoin d'initialiser quoi que ce soit, il suffit de demander à `uv` de tout construire.

Ouvrez un terminal à la racine du dossier `Cahier-Vacances-2026` (dans vscode, clique droit sur le dossier puis "open in integrated terminal") et tapez :

```shell
uv sync
```

`uv` va télécharger la bonne version de python si elle n'est pas déjà sur votre machine, puis créer un dossier `.venv`, c'est notre environnement ! Certains projets vous demanderont d'ajouter une librairie avec `uv add`, ce sera toujours indiqué dans le notebook.

### Étape 4 : ouvrir les notebooks et travailler

Ouvrez le dossier dans votre éditeur (par exemple vscode avec les extensions Python et Jupyter), puis ouvrez le notebook du projet en cours. En haut à droite, cliquez sur "Select Kernel" et choisissez l'environnement du projet (celui qui mentionne `.venv`).

Dans les notebooks il y a des blocs balisés `### START CODE HERE ###` / `### END CODE HERE ###`, ce sont les blocs à remplir ! Il faut compléter entre les deux lignes. Une fois fini il suffit de lancer la cellule de code, puis de voir si les tests passent.

## Récupérer les projets des semaines suivantes

Un nouveau projet est publié chaque dimanche de l'été. Pour le récupérer, ouvrez un terminal à la racine du dossier et tapez :

```shell
git pull
```

Le nouveau dossier `Projet_XX` apparaît, et vous pouvez vous lancer. Les projets déjà publiés ne sont jamais modifiés, donc votre travail dans les notebooks précédents ne sera pas écrasé et vous ne devriez pas rencontrer de conflit.

*Bonne pratique, et bon été !*

import streamlit as st

import time

import numpy as np

import pandas as pd

from pypdf import PdfReader

from sentence\_transformers import SentenceTransformer

from transformers import pipeline

import os



\# Configuration de la page

st.set\_page\_config(

&#x20;   page\_title="Assistant Hôtel Le Belvédère",

&#x20;   page\_icon="🏨",

&#x20;   layout="wide"

)



st.title("🏨 Assistant virtuel de l'Hôtel Le Belvédère")

st.caption("Basé sur RAG (Retrieval-Augmented Generation)")



\# Barre latérale

with st.sidebar:

&#x20;   st.header("⚙️ Configuration")

&#x20;   st.markdown("---")

&#x20;   st.info("L'assistant répond aux questions sur l'hôtel en s'appuyant sur sa documentation officielle.")

&#x20;   

&#x20;   # Bouton pour recharger

&#x20;   if st.button("🔄 Recharger la documentation"):

&#x20;       st.cache\_data.clear()

&#x20;       st.success("✅ Documentation rechargée !")



\# --- CHARGEMENT DES DONNÉES ---

@st.cache\_data

def load\_documentation():

&#x20;   """Charger et préparer la documentation"""

&#x20;   

&#x20;   # 1. Lire les PDF

&#x20;   pdf\_files = \[

&#x20;       "activites\_et\_evenements.pdf",

&#x20;       "chambres\_et\_tarifs.pdf",

&#x20;       "familles\_animaux\_accessibilite.pdf",

&#x20;       "informations\_pratiques.pdf",

&#x20;       "restaurant\_et\_bien\_etre.pdf"

&#x20;   ]

&#x20;   

&#x20;   documents = \[]

&#x20;   for fichier in pdf\_files:

&#x20;       chemin = os.path.join("data", fichier)

&#x20;       try:

&#x20;           reader = PdfReader(chemin)

&#x20;           texte = ""

&#x20;           for page in reader.pages:

&#x20;               texte += page.extract\_text()

&#x20;           documents.append({

&#x20;               "source": fichier,

&#x20;               "texte": texte,

&#x20;               "title": fichier.replace(".pdf", "").replace("\_", " ").title()

&#x20;           })

&#x20;       except Exception as e:

&#x20;           st.warning(f"⚠️ Erreur lors du chargement de {fichier}: {e}")

&#x20;   

&#x20;   # 2. Créer le DataFrame

&#x20;   pages = pd.DataFrame(documents)

&#x20;   pages\["section"] = "## " + pages\["title"] + "\\n\\n" + pages\["texte"]

&#x20;   

&#x20;   # 3. Charger l'embedder

&#x20;   embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

&#x20;   

&#x20;   # 4. Créer les embeddings

&#x20;   chunk\_embeddings = embedder.encode(

&#x20;       pages\["section"].tolist(),

&#x20;       normalize\_embeddings=True

&#x20;   )

&#x20;   

&#x20;   # 5. Charger le générateur

&#x20;   generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

&#x20;   

&#x20;   return pages, chunk\_embeddings, embedder, generator



\# --- FONCTIONS RAG ---

def search(question, pages, chunk\_embeddings, embedder, top\_k=2):

&#x20;   """Rechercher les rubriques pertinentes"""

&#x20;   question\_embedding = embedder.encode(\[question], normalize\_embeddings=True)\[0]

&#x20;   similarities = chunk\_embeddings @ question\_embedding

&#x20;   top\_indices = np.argsort(-similarities)\[:top\_k]

&#x20;   

&#x20;   results = pages.iloc\[top\_indices].copy()

&#x20;   results\["score"] = similarities\[top\_indices]

&#x20;   return results



def build\_prompt(context, question):

&#x20;   """Construire le prompt pour le LLM"""

&#x20;   role = """Tu es l'assistant virtuel de l'Hôtel Le Belvédère, au bord du lac d'Annecy.



Voici la documentation officielle de l'hôtel :"""

&#x20;   

&#x20;   consigne = """Réponds en une ou deux phrases, uniquement à partir des informations 

de la documentation ci-dessus. Si l'information ne s'y trouve pas, réponds exactement : 

"Je ne sais pas, je vous invite à contacter la réception.""

"""

&#x20;   

&#x20;   return f"{role}\\n\\n{context}\\n\\nQuestion d'un client : {question}\\n\\n{consigne}"



def answer\_question(question, pages, chunk\_embeddings, embedder, generator, top\_k=2):

&#x20;   """Pipeline RAG complet"""

&#x20;   # 1. Recherche

&#x20;   retrieved = search(question, pages, chunk\_embeddings, embedder, top\_k)

&#x20;   

&#x20;   # 2. Contexte

&#x20;   context = "\\n\\n".join(retrieved\["section"].tolist())

&#x20;   

&#x20;   # 3. Prompt

&#x20;   prompt = build\_prompt(context, question)

&#x20;   

&#x20;   # 4. Génération

&#x20;   output = generator(prompt, max\_new\_tokens=100)

&#x20;   answer = output\[0]\['generated\_text']

&#x20;   

&#x20;   # Nettoyer la réponse (enlever le prompt)

&#x20;   if "Question d'un client :" in answer:

&#x20;       answer = answer.split("Question d'un client :")\[0].strip()

&#x20;   

&#x20;   return answer, retrieved



\# --- CHARGEMENT ---

with st.spinner("📚 Chargement de la documentation et des modèles..."):

&#x20;   pages, chunk\_embeddings, embedder, generator = load\_documentation()

&#x20;   st.success("✅ Prêt à répondre !")



\# --- INTERFACE PRINCIPALE ---

st.markdown("---")



\# Questions suggérées

st.subheader("💡 Questions suggérées")

col1, col2 = st.columns(2)



with col1:

&#x20;   if st.button("🕐 Quelle est l'heure du check-in ?"):

&#x20;       st.session\_state.question = "A quelle heure commence le check-in ?"

&#x20;   if st.button("📶 Le wifi est-il gratuit ?"):

&#x20;       st.session\_state.question = "Le wifi est-il gratuit ?"



with col2:

&#x20;   if st.button("💰 Prix d'une chambre Classique ?"):

&#x20;       st.session\_state.question = "Combien coute une chambre Classique en basse saison ?"

&#x20;   if st.button("🐕 Acceptez-vous les chiens ?"):

&#x20;       st.session\_state.question = "Acceptez-vous les animaux domestiques ?"



\# Zone de question personnalisée

st.markdown("---")

st.subheader("✍️ Posez votre question")



question = st.text\_input(

&#x20;   "Votre question :",

&#x20;   placeholder="Exemple: Est-ce que la piscine est chauffée ?",

&#x20;   key="question\_input"

)



\# Bouton pour poser la question

if st.button("💬 Poser la question", type="primary"):

&#x20;   if question:

&#x20;       with st.spinner("🔍 Recherche dans la documentation..."):

&#x20;           try:

&#x20;               # Mesurer le temps

&#x20;               start\_time = time.time()

&#x20;               

&#x20;               # Répondre

&#x20;               answer, sources = answer\_question(

&#x20;                   question, pages, chunk\_embeddings, embedder, generator

&#x20;               )

&#x20;               

&#x20;               elapsed\_time = time.time() - start\_time

&#x20;               

&#x20;               # Afficher la réponse

&#x20;               st.markdown("---")

&#x20;               st.subheader("💬 Réponse")

&#x20;               

&#x20;               # Créer une colonne pour la réponse

&#x20;               col1, col2 = st.columns(\[3, 1])

&#x20;               

&#x20;               with col1:

&#x20;                   st.success(answer)

&#x20;               

&#x20;               with col2:

&#x20;                   st.metric("⏱️ Temps", f"{elapsed\_time:.1f}s")

&#x20;               

&#x20;               # Afficher les sources

&#x20;               st.subheader("📚 Sources")

&#x20;               for i, row in sources.iterrows():

&#x20;                   st.info(f"📄 \*\*{row\['title']}\*\* (score: {row\['score']:.3f})")

&#x20;                   

&#x20;                   # Afficher un extrait du texte

&#x20;                   with st.expander(f"Afficher l'extrait de {row\['title']}"):

&#x20;                       st.text(row\['texte']\[:500] + "...")

&#x20;               

&#x20;           except Exception as e:

&#x20;               st.error(f"❌ Erreur : {e}")

&#x20;   else:

&#x20;       st.warning("⚠️ Veuillez saisir une question.")



\# --- INFORMATION SUR LES MODÈLES ---

with st.expander("ℹ️ Informations sur l'assistant"):

&#x20;   st.write("""

&#x20;   \*\*Comment ça fonctionne ?\*\*

&#x20;   

&#x20;   1. 🔍 \*\*Recherche\*\* : Votre question est comparée à la documentation pour trouver les rubriques les plus pertinentes (embeddings).

&#x20;   2. 🧠 \*\*Contexte\*\* : Seules les rubriques pertinentes sont envoyées au modèle.

&#x20;   3. 💬 \*\*Génération\*\* : Le modèle génère une réponse basée sur la documentation.

&#x20;   4. ✋ \*\*Honnêteté\*\* : Si l'information n'est pas dans la documentation, l'assistant dit "Je ne sais pas".

&#x20;   

&#x20;   \*\*Modèles utilisés :\*\*

&#x20;   - Embedding : `paraphrase-multilingual-MiniLM-L12-v2`

&#x20;   - LLM : `Qwen/Qwen2.5-0.5B-Instruct`

&#x20;   """)



\# --- FOOTER ---

st.markdown("---")

st.caption("Projet réalisé avec ❤️ | Hôtel Le Belvédère")


