import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Charger les variables d'environnement
load_dotenv()

# Initialiser FastAPI
app = FastAPI(title="My Eliza Agent")

# Modèle de données pour les requêtes
class Query(BaseModel):
    text: str

# Initialiser le modèle de langage
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="gpt-3.5-turbo"
)

# Créer un template de prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un agent AI basé sur elizaOS. Tu aides à analyser des données et à générer des insights pertinents."),
    ("user", "{input}")
])

# Créer la chaîne de traitement
chain = prompt | llm

@app.post("/query")
async def process_query(query: Query):
    try:
        # Traiter la requête
        response = chain.invoke({"input": query.text})
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur My Eliza Agent!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
