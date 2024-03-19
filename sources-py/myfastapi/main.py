from typing import Union, Annotated
from fastapi import FastAPI
from fastapi import File, UploadFile
import pathlib
import pickle

app = FastAPI()

repo_path = pathlib.Path(__file__).parent.parent.parent.resolve()
pipe = pickle.load(open(repo_path.joinpath("outputs/models/reviewModelPipe.pkl"), "rb"))

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/ai/check-review")
def read_item(query: str):
    moodPredict = pipe.predict_proba([query])[0][1]
    return {"review": query, "mood": moodPredict}

# upload files
# curl -F "file=@orig.png" http://127.0.0.1:8000/imgs/create
@app.post("/imgs/create")
def create_img(file: Annotated[bytes, File()]):
    with open(repo_path.joinpath("temp/my_file.txt"), "wb") as binary_file:
        binary_file.write(file)
    return {"file_size": len(file)}

# upload files
# curl -F "file=@orig.png" http://127.0.0.1:8000/imgs/upload
@app.post("/imgs/upload")
def create_img(file: UploadFile):
    with open(repo_path.joinpath("temp/my_file.txt"), "wb") as binary_file:
        binary_file.write(file.file.read())
    return {"filename": file.filename}


# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run("main:app", port=8000, log_level="info")