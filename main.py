from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import uvicorn
from fastapi.encoders import jsonable_encoder
from functools import lru_cache


app = FastAPI()

class Article(BaseModel):
    content: str

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'\d+', '[NUM]', text)
    return text

@lru_cache(maxsize=1000)
def get_embedding(text):
    return model.encode([text], show_progress_bar=False)[0]

def compare_topic_similarity(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def read_from_file():
    with open('summary_content1.json') as f:
        data = json.load(f)
    return [{"content": article['content'], "title": article['title_en']} for article in data]

# Load model
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)
database = read_from_file()

database_embeddings = [get_embedding(preprocess_text(article['content'])) for article in database]


@app.post("/similarity")
async def compare_article(article: Article):
    start_time = time.time()
    
    preprocessed_article = preprocess_text(article.content)
    article_embedding = get_embedding(preprocessed_article)

    results = []
    for i, db_embedding in enumerate(database_embeddings):
        similarity = compare_topic_similarity(article_embedding, db_embedding)
        if similarity > 0.6:
            match_found = True
            results.append({"article_id": i+1, "similarity": similarity, "title": db_article['title']})
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    end_time = time.time()
    processing_time = end_time - start_time
    data = results[:5]
    json_data = [{'article_id': i['article_id'], 'similarity': float(i['similarity']),'title': i['title']} for i in data]
    
    return {
        "match_found": bool(results),
        "results": json_data,
        "processing_time": processing_time
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)