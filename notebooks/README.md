# Agentic AI Ticket System

## Project Overview
An AI-powered customer support ticket system that retrieves similar historical customer issues and suggests relevant support responses.

## Dataset
- Customer-support pairs: 23,823
- Source domain: Spotify customer support

## Features
- Customer issue input
- Semantic similarity search
- Retrieval-Augmented Generation (RAG)
- Support response suggestion
- Confidence classification
- Automatic ticket resolution
- Ticket creation
- FastAPI REST API

## API Testing Results
- Total tests: 8
- Successful requests: 8
- Failed requests: 0
- High-confidence results: 7
- Automatically resolved tickets: 7

## Tech Stack
- Python
- Pandas
- Sentence Transformers
- FAISS
- FastAPI
- Uvicorn

## API Endpoint

### Create Ticket
POST /create-ticket

Example request:

{
    "customer_issue": "Spotify stops playing after a few songs"
}

## Project Output
The system finds similar historical issues using semantic search and returns a suggested support response with confidence, action, and ticket status.

## Testing
The API was tested with 8 customer issues, with 8 successful requests.
