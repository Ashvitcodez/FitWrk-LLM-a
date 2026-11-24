from fastapi import FastAPI, Header, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_query import FieldFilter

load_dotenv()

db = firestore.Client.from_service_account_json("firestore.json")

model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
embedding_size = 1024

class EmbedModel(BaseModel):
    user_id: str

class ChatModel(BaseModel):
    content: str
    thread_id: str | None = None

def verify_api_key(x_api_key : str = Header(None)):
    api_KEY = os.getenv("API_KEY")
    if not x_api_key or x_api_key != api_KEY:
        raise HTTPException(status_code = 401, detail = "invalid or missing API key")

@app.get("/")
def hello():
    return "application is running"

@app.get("/health", dependencies = [Depends(verify_api_key)])
def status():
    return "I am alive!"

@app.post("/assistant/fitness/createThread")
def defaultThreadCreateFitness():
    thread = assistant.beta.threads.create()
    thread_id = thread.id
    return thread_id

@app.post("/assisstant/fitness", dependencies = [Depends(verify_api_key)])
def fitnessAssistant(chatModel:ChatModel):
    assistantID = os.getenv("ASSISTANT_ID")
    if chatModel.thread_id:
        thread_id = chatModel.thread_id
    else:
        thread = assistant.beta.threads.create()
        thread_id = thread.id
    message = assistant.beta.threads.messages.create(thread_id = thread_id, role = "user", content = chatModel.content)
    run = assistant.beta.threads.runs.create(thread_id = thread_id, assistant_id = assistantID)
    while True:
        status = assistant.beta.threads.runs.retrieve(thread_id = thread_id, run_id = run.id)
        if status.status == "completed":
            break
        time.sleep(1)
    messages = assistant.beta.threads.messages.list(thread_id = thread_id)
    messages = [ [message.role, message.content[0].text.value] for message in messages.data]
    return {"thread_id" : thread_id, "messages" : messages}

@app.post("/assisstant/nutritionist/createThread")
def defualtThreadNutritionist():
    thread = assistant.beta.threads.create()
    thread_id = thread.id
    return thread_id

@app.post("/assisstant/nutritionist", dependencies = [Depends(verify_api_key)])
def nutritionist(chatModel:ChatModel):
    assistantID = os.getenv("ASSISTANT_ID")
    if chatModel.thread_id:
        thread_id = chatModel.thread_id
    else:
        thread = assistant.beta.threads.create()
        thread_id = thread.id
    message = assistant.beta.threads.messages.create(thread_id = thread_id, role = "user", content = chatModel.content)
    run = assistant.beta.threads.runs.create(thread_id = thread_id, assistant_id = assistantID)
    while True:
        status = assistant.beta.threads.runs.retrieve(thread_id = thread_id, run_id = run.id)
        if status.status == "completed":
            break
        time.sleep(1)
    messages = assistant.beta.threads.messages.list(thread_id = thread_id)
    messages = [ [message.role, message.content[0].text.value] for message in messages.data]
    return {"thread_id" : thread_id, "messages" : messages}

def dictionaryToText(diction):
    """Convert user profile dictionary to text for embedding"""
    # Create a copy to avoid modifying original
    profile_copy = dict(diction)

    # Remove fields that shouldn't be in embedding
    fields_to_remove = ["createdAt", "email", "uid", "profileComplete", "photoURL"]
    for field in fields_to_remove:
        profile_copy.pop(field, None)

    txts = []
    for key, value in profile_copy.items():
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        txts.append(f"{key}: {value}")

    return " | ".join(txts)

def calcEmbed(user_id):
    """Calculate and store embedding for a user profile"""
    try:
        profileRef = db.collection("users").document(user_id)
        profile = profileRef.get().to_dict()

        if not profile:
            return {"error": "User profile not found", "user_id": user_id}

        # Convert profile to text
        newProfTxt = dictionaryToText(profile)
        print(f"Profile text for {user_id}: {newProfTxt}")

        # Generate embedding
        new_vector = model.encode(newProfTxt, normalize_embeddings=True)

        # Update Firestore with embedding
        profileRef.update({"user_embedding": Vector(new_vector.tolist())})

        return {"message": "created vector!", "user_id": user_id}

    except Exception as e:
        print(f"Error creating embedding for {user_id}: {e}")
        return {"error": str(e), "user_id": user_id}

@app.post("/discover/embedding")
def provEmbedding(embedModel: EmbedModel):
    """Generate embedding for a single user"""
    return calcEmbed(embedModel.user_id)

@app.get("/discover/allUserEmbeddings")
def calcEmbeddingForExistingUsers():
    """Calculate embeddings for all users who don't have one yet"""
    try:
        # Get all users
        users_ref = db.collection("users")
        users = users_ref.stream()

        success_count = 0
        error_count = 0

        for user in users:
            user_data = user.to_dict()

            # Only create embedding if user doesn't have one
            if "user_embedding" not in user_data:
                result = calcEmbed(user.id)
                if "error" in result:
                    error_count += 1
                else:
                    success_count += 1

        return {
            "message": "Completed embedding generation",
            "success": success_count,
            "errors": error_count
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/discover/getMatchCompatability")
def getMatch(embedModel: EmbedModel):
    """
    Get compatible matches for a user based on vector similarity.
    Returns list of users with their cosine distance.
    Lower distance = more similar (better match).
    """
    try:
        # ✅ FIX: Use embedModel.user_id instead of undefined user_id
        profileRef = db.collection("users").document(embedModel.user_id).get()

        if not profileRef.exists:
            raise HTTPException(status_code=404, detail="User profile not found")

        # Get user's embedding
        targetEmbed = profileRef.get("user_embedding")

        if not targetEmbed:
            raise HTTPException(
                status_code=400,
                detail="User has no embedding. Generate embedding first at /discover/embedding"
            )

        # Create query vector
        queryVector = Vector(targetEmbed)

        # ✅ FIX: Increase limit to 50 for better matching
        # Frontend will filter by threshold
        vectorQuery = db.collection("users").find_nearest(
            vector_field="user_embedding",
            query_vector=queryVector,
            distance_measure="COSINE",
            limit=50
        )

        # ✅ FIX: Changed vector_query to vectorQuery
        results = vectorQuery.get()
        matches = []

        for document in results:
            # ✅ FIX: Use embedModel.user_id instead of embedModel.id
            if document.id == embedModel.user_id:
                continue  # Skip the user themselves

            # Get distance from metadata
            distance = document.get("distance")

            if distance is not None:
                matches.append({
                    "user_id": document.id,
                    "distance": float(distance)  # Ensure it's a float
                })

        print(f"Found {len(matches)} potential matches for user {embedModel.user_id}")
        return matches

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in getMatchCompatability: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check for testing
@app.get("/api/test")
def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {
        "status": "ok",
        "message": "FitWrk backend is running!",
        "endpoints": {
            "health": "/health",
            "chat_fitness": "/assisstant/fitness",
            "chat_nutrition": "/assisstant/nutritionist",
            "generate_embedding": "/discover/embedding",
            "get_matches": "/discover/getMatchCompatability"
        }
    }
