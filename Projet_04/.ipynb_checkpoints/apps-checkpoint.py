import streamlit as st
import time
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# Configuration de la page
st.set_page_config(
    page_title="Assistant Hôtel Le Belvédère",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Assistant virtuel de l'Hôtel Le Belvédère")
st.caption("Basé sur RAG (Retrieval-Augmented Generation)")

# Barre latérale
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    st.info("L'assistant répond aux questions sur l'hôtel en s'appuyant sur sa documentation officielle.")
    
    # Bouton pour recharger
    if st.button("🔄 Recharger la documentation"):
        st.cache_data.clear()
        st.success("✅ Documentation rechargée !")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_documentation():
    """Charger et préparer la documentation"""
    
    # 1. Lire les PDF
    pdf_files = [
        "activites_et_evenements.pdf",
        "chambres_et_tarifs.pdf",
        "familles_animaux_accessibilite.pdf",
        "informations_pratiques.pdf",
        "restaurant_et_bien_etre.pdf"
    ]
    
    documents = []
    for fichier in pdf_files:
        chemin = os.path.join("data", fichier)
        try:
            reader = PdfReader(chemin)
            texte = ""
            for page in reader.pages:
                texte += page.extract_text()
            documents.append({
                "source": fichier,
                "texte": texte,
                "title": fichier.replace(".pdf", "").replace("_", " ").title()
            })
        except Exception as e:
            st.warning(f"⚠️ Erreur lors du chargement de {fichier}: {e}")
    
    # 2. Créer le DataFrame
    pages = pd.DataFrame(documents)
    pages["section"] = "## " + pages["title"] + "\n\n" + pages["texte"]
    
    # 3. Charger l'embedder
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # 4. Créer les embeddings
    chunk_embeddings = embedder.encode(
        pages["section"].tolist(),
        normalize_embeddings=True
    )
    
    # 5. Charger le générateur
    generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    
    return pages, chunk_embeddings, embedder, generator

# --- FONCTIONS RAG ---
def search(question, pages, chunk_embeddings, embedder, top_k=2):
    """Rechercher les rubriques pertinentes"""
    question_embedding = embedder.encode([question], normalize_embeddings=True)[0]
    similarities = chunk_embeddings @ question_embedding
    top_indices = np.argsort(-similarities)[:top_k]
    
    results = pages.iloc[top_indices].copy()
    results["score"] = similarities[top_indices]
    return results

def build_prompt(context, question):
    """Construire le prompt pour le LLM"""
    role = """Tu es l'assistant virtuel de l'Hôtel Le Belvédère, au bord du lac d'Annecy.

Voici la documentation officielle de l'hôtel :"""
    
    consigne = """Réponds en une ou deux phrases, uniquement à partir des informations 
de la documentation ci-dessus. Si l'information ne s'y trouve pas, réponds exactement : 
"Je ne sais pas, je vous invite à contacter la réception.""
"""
    
    return f"{role}\n\n{context}\n\nQuestion d'un client : {question}\n\n{consigne}"

def answer_question(question, pages, chunk_embeddings, embedder, generator, top_k=2):
    """Pipeline RAG complet"""
    # 1. Recherche
    retrieved = search(question, pages, chunk_embeddings, embedder, top_k)
    
    # 2. Contexte
    context = "\n\n".join(retrieved["section"].tolist())
    
    # 3. Prompt
    prompt = build_prompt(context, question)
    
    # 4. Génération
    output = generator(prompt, max_new_tokens=100)
    answer = output[0]['generated_text']
    
    # Nettoyer la réponse (enlever le prompt)
    if "Question d'un client :" in answer:
        answer = answer.split("Question d'un client :")[0].strip()
    
    return answer, retrieved

# --- CHARGEMENT ---
with st.spinner("📚 Chargement de la documentation et des modèles..."):
    pages, chunk_embeddings, embedder, generator = load_documentation()
    st.success("✅ Prêt à répondre !")

# --- INTERFACE PRINCIPALE ---
st.markdown("---")

# Questions suggérées
st.subheader("💡 Questions suggérées")
col1, col2 = st.columns(2)

with col1:
    if st.button("🕐 Quelle est l'heure du check-in ?"):
        st.session_state.question = "A quelle heure commence le check-in ?"
    if st.button("📶 Le wifi est-il gratuit ?"):
        st.session_state.question = "Le wifi est-il gratuit ?"

with col2:
    if st.button("💰 Prix d'une chambre Classique ?"):
        st.session_state.question = "Combien coute une chambre Classique en basse saison ?"
    if st.button("🐕 Acceptez-vous les chiens ?"):
        st.session_state.question = "Acceptez-vous les animaux domestiques ?"

# Zone de question personnalisée
st.markdown("---")
st.subheader("✍️ Posez votre question")

question = st.text_input(
    "Votre question :",
    placeholder="Exemple: Est-ce que la piscine est chauffée ?",
    key="question_input"
)

# Bouton pour poser la question
if st.button("💬 Poser la question", type="primary"):
    if question:
        with st.spinner("🔍 Recherche dans la documentation..."):
            try:
                # Mesurer le temps
                start_time = time.time()
                
                # Répondre
                answer, sources = answer_question(
                    question, pages, chunk_embeddings, embedder, generator
                )
                
                elapsed_time = time.time() - start_time
                
                # Afficher la réponse
                st.markdown("---")
                st.subheader("💬 Réponse")
                
                # Créer une colonne pour la réponse
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.success(answer)
                
                with col2:
                    st.metric("⏱️ Temps", f"{elapsed_time:.1f}s")
                
                # Afficher les sources
                st.subheader("📚 Sources")
                for i, row in sources.iterrows():
                    st.info(f"📄 **{row['title']}** (score: {row['score']:.3f})")
                    
                    # Afficher un extrait du texte
                    with st.expander(f"Afficher l'extrait de {row['title']}"):
                        st.text(row['texte'][:500] + "...")
                
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
    else:
        st.warning("⚠️ Veuillez saisir une question.")

# --- INFORMATION SUR LES MODÈLES ---
with st.expander("ℹ️ Informations sur l'assistant"):
    st.write("""
    **Comment ça fonctionne ?**
    
    1. 🔍 **Recherche** : Votre question est comparée à la documentation pour trouver les rubriques les plus pertinentes (embeddings).
    2. 🧠 **Contexte** : Seules les rubriques pertinentes sont envoyées au modèle.
    3. 💬 **Génération** : Le modèle génère une réponse basée sur la documentation.
    4. ✋ **Honnêteté** : Si l'information n'est pas dans la documentation, l'assistant dit "Je ne sais pas".
    
    **Modèles utilisés :**
    - Embedding : `paraphrase-multilingual-MiniLM-L12-v2`
    - LLM : `Qwen/Qwen2.5-0.5B-Instruct`
    """)

# --- FOOTER ---
st.markdown("---")
st.caption("Projet réalisé avec ❤️ | Hôtel Le Belvédère")