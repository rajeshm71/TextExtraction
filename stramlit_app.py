import re
import numpy as np
import pandas as pd
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import nltk
import streamlit as st
from streamlit.web import cli as stcli
import sys
# nltk.download('punkt')
# from transformers import pipeline

model_checkpoint = "bert_finetuned_ner/checkpoint-118380"
def preprocess(df, column='review'):
    if column == 'review':
        # drop duplicate rows
        df = df.drop_duplicates()

        # drop rows where na values present in parts column
        if 'parts' in df.columns:
            df = df.dropna(subset=['parts'])

    # lowercase the text
    df[column] = df[column].str.lower()

    # add septoken
    if 'parts' in df.columns and column == 'parts':
        df[column] = df[column].apply(
            lambda x: str(x).replace("...", " septoken ") if '...' in str(x) else str(x) + " septoken ")

    # remove special characters and numbers
    df[column] = df[column].apply(lambda x: re.sub(r'[^\w\s]+', '', str(x)))

    # remove underscores
    df[column] = df[column].apply(lambda x: x.replace("_", "") if '_' in x else x)

    # tokenize the reviews
    df[column] = df[column].apply(lambda x: word_tokenize(x))

    return df

# Function to remove special characters and spaces from a string
def clean_string(string):
    cleaned_string = re.sub(r'[^\w]', '', string) # Removes special characters and spaces including underscores
    return cleaned_string

def get_processed_sentence(sentence, label):
    # Create a dataframe from the sentence and label inputs
    test_df = pd.DataFrame({'review':[sentence], 'label':[label]})
    # Preprocess the dataframe
    test_df = preprocess(test_df)
    # Join the review column in the dataframe into a single string
    sentence = " ".join(test_df['review'][0])
    # Clean the label column in the dataframe
    label = clean_string(test_df['label'][0])
    # Concatenate the processed sentence and label into a single string
    sentence = sentence + " " + label
    # Return the combined string
    return sentence


def get_predictions_sentence(sentence, label, model_checkpoint):
    # Initialize an empty string to store the processed sentence
    sent = ''

    # Get the processed sentence by calling the get_processed_sentence function
    sentence = get_processed_sentence(sentence, label)

    # Initialize a token-classification pipeline using the specified model checkpoint
    token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="simple")

    # Get the predictions from the pipeline on the processed sentence
    predictions = token_classifier(sentence)

    # Loop through the predicted entities in the sentence
    for entities in predictions:
        # Check if there are any entities in the current prediction
        if len(entities) > 0:
            # Get the word and score of the current entity
            pred_sent = entities['word']
            score = entities['score']
            # Check if the score is higher than 0.85
            if score > 0.85:
                # Add the predicted word to the final sentence
                sent = sent + pred_sent + '...'
    return sent

def run():
    st.title('Text Extraction App')
    st.write('Welcome to My Text Extraction App!')

    form = st.form('text_extraction')
    sentence = form.text_area('Enter your review')
    label = form.text_area('Enter your label')
    Predict = form.form_submit_button('Submit')

    prediction = 'Make Prediction'
    if Predict:
        prediction = get_predictions_sentence(sentence, label, model_checkpoint)

    if prediction:
        st.success(f'{prediction} ')
    else:
        st.error(f'No Part Found')

if __name__ == '__main__':
    try:
        sys.argv = ["streamlit", "run", "stramlit_app.py"]
        sys.exit(stcli.main())
        run()
    except RuntimeError:
        print('yes')


