## How to run
use it multiple times until all installed right, it's skips some dependencies on the first-second runs  
`pip install -r reqs.txt`  

after installation go to `sources-py` folder and create there `.venv`  
then open notebooks an run all code blocks in them:  
1. `sources-py/notebooks/import-data.ipynb` imports and prepares data from dataset  
2. `sources-py/notebooks/build-reviewModel.ipynb` inits model and learns it with imported data, then ouputs the model  
3. `sources-py/notebooks/run-models` just runs builded model on some set of data to check it works  

after that model is ready and we need to run web-server to get access to that model  

to run web-server simply call `uvicorn myfastapi.main:app` (using pythons `.venv` ofcourse)  
this will host an http web server uvicorn with fastapi application running on it

you can check server by openning `http://127.0.0.1:8000` in your browser (or with curl)  
in `sources-py/myfastapi/main.py` you can see some endpoints  
to checkout autogenerated docs for this app - open `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/redoc`  

to check image uploading you can call  
`curl -F "file=@datasets/orig.png" http://127.0.0.1:8000/imgs/upload`  
server will receive image and store it in `temp` folder  

---
### Instruments

- pip - for package management
- pipdeptree - for formatting reqs.txt

- ipykernel - jupiter notebook for snippeting and preparing models
- pickleshare - to use %store to share vars between notebooks 
- pandas + openpyxl - to parse scv and excel datasets

- matplotlib - for diagrams
- tensorflow & scikit-learn - for model learning

- uvicorn - for web hosting
- fastapi - for building web app
- python-multipart - for file uploading


- OpenCV


---
### List of Terms

- AI - Artificial Intelligence
- ML - Machine Learning
- DL - Deep Learning
- NLP - Natutal Language Processing
- CV - Computer Vision