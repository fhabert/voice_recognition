import openai
from transformers import pipeline 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import spacy
import logging
import speech

input_url = "https://plato.stanford.edu/ENTRIES/church-turing/"

def get_text_search(query: str, max_words: int) -> str:
    input_req = f"https://plato.stanford.edu/search/searcher.py?query={query}"
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    executable_p = "C:\Program Files\Google\Chrome\chromedriver.exe"
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('selenium.webdriver.remote.remote_connection').setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    driver = webdriver.Chrome(executable_path=executable_p, options=chrome_options)
    driver.get(input_req)
    search_results = driver.find_element(By.CLASS_NAME, "search_results")
    inner_childs = search_results.find_elements(By.CLASS_NAME, "result_listing")
    if len(inner_childs) != 0:
        el = inner_childs[0].find_element(By.CLASS_NAME, "result_title")
        link = el.find_element(By.CLASS_NAME, "l")
        link.click()
    else:
        raise Exception("Your request has zero results")
    content = driver.find_element(By.ID, "content").text
    words = content.split()[:max_words]
    limited = ' '.join(words)
    return limited

def get_summary(text: str):
    print("Let's get you a summary about your query")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=250, min_length=100, do_sample=False)
    return summary[0]["summary_text"]

def audio_to_text(seconds: int):
    _ = speech.record_something(seconds)
    audio_file = open("./audio.wav", "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)
    print("Query:", transcript["text"])
    return transcript["text"]

def pass_text_to_model(prompt_text: str) -> str:
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role": "user", "content": prompt_text }])
    transcription = completion["choices"][0]['message']["content"]
    return transcription

def get_features(query: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(query)
    print("Selecting features..")
    if len(ner_results) == 1:
        print("Key:", ner_results[0]["word"])
        return ner_results[0]["word"]
    elif len(ner_results) > 1:
        dict_class = { "LOC": [], "ORG": [], "PER": [], "MISC": [] }
        for item in ner_results:
            dict_class[item["entity"][2:]].append(item["word"])
        if len(dict_class["PER"]) > 0:
            print("Selecting features..")
            print("Key:", " ".join(dict_class["PER"]))
            return " ".join(dict_class["PER"])
    else:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ != "PRON":
                noun_phrases.append(chunk.text)
        print(noun_phrases)
        return noun_phrases[0]
    

def continue_query(text: str):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt")
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id


print("Tell us your query \n")
request = audio_to_text(3)
features = get_features(request)
max_words = 1500
text_to_search = get_text_search(features, max_words)
short_text = get_summary(text_to_search)
print(short_text)
print("New query?\n", 10 * "-")
request = audio_to_text(5)
next = continue_query(request)
print("Next", next)
if next == 1:
    while next == 1:
        print("What is your new query? \n")
        request = audio_to_text(5)
        ans = pass_text_to_model(request)
        print(ans)
        print("New query?\n", 10 * "-")
        request = audio_to_text(5)
        next = continue_query(request)
else:
    print("Ok see you soon!!")


