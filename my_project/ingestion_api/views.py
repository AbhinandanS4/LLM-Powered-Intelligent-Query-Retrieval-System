from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
import os
import sys
import uuid

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http import models
from groq import Groq
from dotenv import load_dotenv

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DOTENV_PATH = BASE_DIR / '.env'

load_dotenv(dotenv_path=DOTENV_PATH)
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    # This will stop the server from starting and give a clear error
    raise ValueError("FATAL ERROR: GROQ_API_KEY environment variable not set.") 

try:
    # Initialize the client. This will only run if the key exists.
    groq_client = Groq(api_key=groq_api_key) 
except Exception as e:
    # Handle other potential errors from the Groq library itself
    print(f"Failed to initialize Groq client: {e}", file=sys.stderr)
    raise e

# --- Configuration ---
COLLECTION_NAME = "policy_documents"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VECTOR_SIZE = 384 # Dimension for 'all-MiniLM-L6-v2'

# --- Initialize Global Objects ---
encoder = None
qdrant_db_client = None

try:
    # Load the embedding model
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    qdrant_client = qdrant_client.QdrantClient(":memory:")
    print("Qdrant client initialized in-memory.")
    
    # Create the collection every time the server starts, as it's not persistent.
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )
    print(f"In-memory collection '{COLLECTION_NAME}' created.")

except Exception as e:
    print(f"An error occurred during initialization: {e}", file=sys.stderr)
    encoder = None
    qdrant_client = None


@csrf_exempt
@require_POST
def upload_document_view(request):
    """API endpoint to upload various documents (PDF, DOCX, EML), process them, and index them."""
    if not encoder or not qdrant_client:
        return JsonResponse({"status": "error", "message": "Backend services not initialized."}, status=503)

    if 'document' not in request.FILES:
        return JsonResponse({"status": "error", "message": "No document provided."}, status=400)

    doc_file = request.FILES['document']
    
    # Use 'unstructured' to handle multiple document types
    try:
        # 'unstructured' can automatically infer the file type
        elements = partition(file=doc_file)
        all_text = "\n\n".join([el.text for el in elements])

        if not all_text:
            return JsonResponse({"status": "error", "message": "Could not extract text from the document."}, status=400)
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"Failed to parse document with unstructured: {e}"}, status=500)

    # --- The rest of your function (text splitting, encoding, upserting) remains the same ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    chunks = text_splitter.split_text(all_text)
    
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # ... (rest of the try/except blocks for encoding and upserting) ...

    return JsonResponse({"status": "success", "message": f"Document '{doc_file.name}' processed successfully.", "chunks_indexed": len(chunks)})
@csrf_exempt
@require_POST
def query_view(request):
    """
    API endpoint to receive a query, find relevant chunks, and use an LLM
    to generate a structured decision based on the retrieved context.
    """
    # --- Step 1: Initial checks and query retrieval (Same as before) ---
    if not encoder or not qdrant_client:
        return JsonResponse({"status": "error", "message": "Backend services not initialized."}, status=503)

    try:
        data = json.loads(request.body)
        query_text = data.get("query")
        if not query_text:
            return JsonResponse({"status": "error", "message": "No query text provided."}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({"status": "error", "message": "Invalid JSON format."}, status=400)

    # --- Step 2: Semantic Search in Qdrant (Same as before) ---
    try:
        query_vector = encoder.encode(query_text).tolist()
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5 # Retrieve the top 5 most relevant chunks
        )
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"Failed to search in Qdrant: {e}"}, status=500)

    context_clauses = [result.payload['text'] for result in search_results]
    context_string = "\n\n---\n\n".join(context_clauses)

    # This prompt is the "brain" of your new feature. It tells the LLM how to behave.
    prompt = f"""
    You are an expert insurance policy analyst. Your task is to evaluate a client's query against a set of retrieved policy clauses and make a final, structured decision.

    **Client Query:**
    "{query_text}"

    **Retrieved Policy Clauses:**
    ---
    {context_string}
    ---

    **Instructions:**
    1.  Carefully analyze the "Client Query" and the "Retrieved Policy Clauses".
    2.  Determine the correct decision (Approved, Rejected, or Needs More Information).
    3.  If a payout amount is mentioned or can be calculated from the clauses, specify it. Otherwise, use null.
    4.  You MUST provide a justification by mapping your decision directly to the specific clause(s) it was based on.
    5.  Your entire response MUST be a single, valid JSON object. Do not include any text or formatting before or after the JSON object.

    **Required JSON Output Format:**
    {{
      "decision": "Approved" | "Rejected" | "Needs More Information",
      "amount": <number> | null,
      "justification": [
        {{
          "clause_text": "<The exact text of the relevant clause from the provided context>",
          "reasoning": "<Your brief explanation of how this specific clause supports the decision>"
        }}
      ]
    }}

    Now, provide your analysis as a single JSON object.
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192", # A fast and capable model
            temperature=0.0, # Set to 0 for deterministic, fact-based output
            response_format={"type": "json_object"}, # 👈 This is the key for reliable JSON!
        )
        response_content = chat_completion.choices[0].message.content
        decision_data = json.loads(response_content)
        
        return JsonResponse(decision_data)

    except json.JSONDecodeError:
        return JsonResponse({
            "status": "error", 
            "message": "Failed to parse the decision from the AI model.",
            "raw_response": response_content # Include raw response for debugging
        }, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"An error occurred with the Groq API: {e}"}, status=500)
