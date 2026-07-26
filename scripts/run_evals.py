import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline
import yaml

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import cleanly from scripts
from scripts.check_guardrails import run_input_guardrail, run_output_guardrail


def calculate_rag_metrics(retrieved_ids: list, ground_truth_ids: list, total_db_size: int):
    retrieved_set = set(retrieved_ids)
    gt_set = set(ground_truth_ids)
    
    tp = len(retrieved_set.intersection(gt_set))
    fp = len(retrieved_set - gt_set)
    fn = len(gt_set - retrieved_set)
    tn = total_db_size - (len(retrieved_set) + fn)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            mrr = 1.0 / rank
            break
            
    return {"precision": precision, "recall": recall, "accuracy": accuracy, "mrr": mrr}

def run_rag_with_evals(file_path: str, query: str, expected_gt_ids: list, top_k: int = 2):
    # 1. INPUT GUARDRAIL (Llama Guard + NeMo)
    if not run_input_guardrail(query):
        return "BLOCKED BY INPUT GUARDRAILS"

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    prompt_file = "prompts/prompt_v1.yaml"
    with open(prompt_file, "r", encoding="utf-8") as pf:
        prompt_config = yaml.safe_load(pf)

    system_instructions = prompt_config.get("system_prompt", "")

    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_text(text)
    total_chunks = len(chunks)
    
    hf_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection(name="pure_python_rag", embedding_function=hf_embeddings)
    
    ids = [f"chunk_{i}" for i in range(total_chunks)]
    collection.add(documents=chunks, ids=ids)
    
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved_ids = results["ids"][0]
    retrieved_texts = results["documents"][0]
    
    generator = pipeline("text-generation", model="gpt2")
    context = " ".join(retrieved_texts)
    prompt = f" {system_instructions} \n\n Context: {context}\nQuestion: {query}\nAnswer:"
    
    llm_output = generator(prompt, max_new_tokens=20, pad_token_id=50256)[0]["generated_text"]

    # 4. OUTPUT GUARDRAIL (Llama Guard + NeMo)
    if not run_output_guardrail(llm_output):
        return "BLOCKED BY OUTPUT GUARDRAILS"
    
    clean_answer = llm_output.replace(prompt, "").strip()
    
    scores = calculate_rag_metrics(retrieved_ids, expected_gt_ids, total_chunks)
    return clean_answer, retrieved_ids, scores

if __name__ == "__main__":
    file_name = "interview_sample.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(
            "Wipro AI Talent Quest evaluates technical architect roles.\n"
            "The core interview focus is multi-agent systems and metrics.\n"
            "Engineering delivery hubs are located in India, US, and Europe.\n"
            "The hybrid work policy requires developers to coordinate schedules."
        )

    user_query = "Where are the engineering hubs located?"
    target_ground_truth = ["chunk_1"] 
    
    answer, fetched_ids, metrics = run_rag_with_evals(file_name, user_query, target_ground_truth)
    
    print("\n================== 📊 EXECUTION RUN METRICS ==================")
    print(f"Retrieved Chunks: {fetched_ids} (Expected Ground Truth: {target_ground_truth})")
    print(f"Precision       : {metrics['precision']}")
    print(f"Recall          : {metrics['recall']}")
    print(f"MRR             : {metrics['mrr']}")
    print("==============================================================")
    
    os.remove(file_name)

    # 🚨 CI/CD QUALITY GATE THRESHOLD 🚨
    if metrics['recall'] < 0.5 or metrics['mrr'] < 0.5:
        print("❌ CI/CD QUALITY GATE FAILED: Retrieval metrics below threshold!")
        sys.exit(1)
    else:
        print("✅ CI/CD QUALITY GATE PASSED: Metrics meet production standards.")
        sys.exit(0)