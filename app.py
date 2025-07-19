from fastapi import FastAPI, Query
from rag_bot import qa_chain

app = FastAPI()

@app.get("/ask")
async def ask_question(q: str = Query(..., description="Your question")):
    response = qa_chain.invoke(q)  # use .invoke() instead of .run()
    return {"question": q, "answer": response}
