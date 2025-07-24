import os
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"version": os.getenv("VERSION")}