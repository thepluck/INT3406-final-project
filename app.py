import os
from subprocess import PIPE, Popen
import streamlit as st
from transformers import pipeline
import sys

current_dir = os.getcwd()

vietsenti = Popen(
    [f"{sys.executable}", "vietsenti.py"],
    cwd=current_dir + "/VietSentiWordNet/",
    stdin=PIPE,
    stdout=PIPE,
    stderr=PIPE,
)

segment = Popen(
    [f"{sys.executable}", "segment.py"], stdin=PIPE, stdout=PIPE, stderr=PIPE
)

### Ignore the first 34 lines
for i in range(0, 34):
    ignore = vietsenti.stdout.readline()

ignore = segment.stdout.readline()


def word_segment(text):
    input = text + "\n"
    segment.stdin.write(input.encode())
    segment.stdin.flush()

    return segment.stdout.readline().decode()


def get_sentiment_words(text):
    input = text + "\n"
    vietsenti.stdin.write(input.encode())
    vietsenti.stdin.flush()

    output = vietsenti.stdout.readline()
    output = vietsenti.stdout.readline()
    output = vietsenti.stdout.readline()
    output = vietsenti.stdout.readline()
    output = vietsenti.stdout.readline()

    ### Extract sentiment word
    output = output.decode()
    first = output.find("Sentiment words: ")
    last = output.find("SentencScore", first + 16)
    sentiment_word = output[first + 16 : last - 5]
    return sentiment_word.split(",")

def print_label(output):
    def mapping(label):
        if label == "LABEL_1":
            return "Negative"
        else:
            return "Positive"
    
    st.write(mapping(output[0]["label"]), "with score", output[0]["score"])
    st.write(mapping(output[1]["label"]), "with score", output[1]["score"])

classifier = pipeline(
    task="text-classification",
    model="./SentimentPhoBERT-LoRA-256",
    tokenizer="./SentimentPhoBERT-LoRA-256",
)

with st.form(key="my_form"):
    text = st.text_input(label="Enter your text here")
    submit_button = st.form_submit_button(label="Submit")
    if submit_button:
        text = word_segment(text)
        sentiment_words = get_sentiment_words(text)
        print_label(classifier(text, top_k=None))
        st.write("Sentiment words:", ", ".join(sentiment_words))



vietsenti.stdin.close()
vietsenti.terminate()
vietsenti.wait(timeout=0.2)
segment.stdin.close()
segment.terminate()
segment.wait(timeout=0.2)

print("Done")
