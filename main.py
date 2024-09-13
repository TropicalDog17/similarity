from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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

# Define a global variable to store the last updated time of the recent n articles
last_updated_time = time.time()

class Article(BaseModel):
    content: str

class SimilarityRequest(Article):
    threshold: float = Field(ge=0, le=1, default=0.8)

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

# pre-calculate the embeddings for all recent n articles.
database_embeddings = [get_embedding(preprocess_text(article['content'])) for article in database]


@app.get("/similarity")
async def compare_article(req: SimilarityRequest):
    start_time = time.time()
    
    preprocessed_article = preprocess_text(req.content)
    article_embedding = get_embedding(preprocessed_article)

    results = []
    for i, db_embedding in enumerate(database_embeddings):
        similarity = compare_topic_similarity(article_embedding, db_embedding)
        # TODO: Define a threshold for similarity
        if similarity > req.threshold:
            match_found = True
            results.append({"article_id": i+1, "similarity": similarity, "title": database[i]['title']})
    
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

@app.get("last_updated")
async def last_updated():
    return {"last_updated": last_updated_time}

@app.post("/recent")
async def update_recent_articles(new_article: Article):
    # TODO: fix global variable
    global last_updated_time
    last_updated_time = time.time()
    new_article_embedding = get_embedding(preprocess_text(new_article.content))

    # add new article and remove the oldest from the recent n articles
    database.append(new_article)
    database = database[1:]
    database_embeddings.append(new_article_embedding)
    database_embeddings = database_embeddings[1:]
    return {
        "message": "Recent articles updated successfully",
        "last_updated": last_updated_time
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)