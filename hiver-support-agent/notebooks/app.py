
import pandas as pd
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from datetime import datetime
import uuid

app = FastAPI(title="Agentic AI Ticket System")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("spotify_faiss_index.faiss")

ai_dataset = pd.read_pickle("spotify_ai_dataset.pkl")


class TicketRequest(BaseModel):
    customer_issue: str


def search_similar_issues(query, k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for idx, distance in zip(indices[0], distances[0]):
        results.append({
            "customer_issue": ai_dataset.iloc[idx]["customer_issue"],
            "support_response": ai_dataset.iloc[idx]["support_response"],
            "distance": float(distance)
        })

    return results


@app.get("/")
def home():
    return {
        "message": "Agentic AI Ticket System API is running"
    }


@app.post("/create-ticket")
def create_ticket(request: TicketRequest):

    results = search_similar_issues(
        request.customer_issue,
        k=3
    )

    best_result = results[0]
    distance = best_result["distance"]

    if distance < 0.4:
        confidence = "High"
        action = "Auto Respond"
        status = "Resolved Automatically"

    elif distance < 0.7:
        confidence = "Medium"
        action = "Human Review"
        status = "Needs Review"

    else:
        confidence = "Low"
        action = "Escalate to Support Agent"
        status = "Escalated"

    return {
        "ticket_id": str(uuid.uuid4())[:8],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_issue": request.customer_issue,
        "suggested_response": best_result["support_response"],
        "confidence": confidence,
        "distance": round(distance, 4),
        "action": action,
        "status": status
    }
