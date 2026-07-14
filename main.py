import os
import asyncio
from fastapi import FastAPI, Request

app = FastAPI(title="FAISS 벡터DB 서버")

@app.get('/')
async def read_root():
    return {'message': 'server is ready'}