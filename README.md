# Friend Script Generator

This repo aims to train a simple transformer-based model on F.R.I.E.N.D.S TV show transcript, in order to generate text similar to it.

## How to run?

First, download the dataset from "https://www.kaggle.com/datasets/divyansh22/friends-tv-show-script", unzip it and put the "Friends_Transcript.txt" file on "./src/data/Friends_Transcript.txt" path.

Then, add the `src` folder to the `PYTHONPATH`:
```
export PYTHONPATH=${PWD}
```

Then install the dependencies:
```
pip install -r requirements.txt
```
Finaly, train and generate new script using:
```
python src/run.py
```
