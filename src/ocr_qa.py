import os
import asyncio
import requests
from PIL import Image
import streamlit as st
import pytesseract as ocr
from dotenv import load_dotenv

load_dotenv()

# Replace 'YOUR_BEARER_TOKEN' with your actual Bearer token
bearer_token = os.getenv('HUGGING_FACE_TOKEN')

url = 'https://api-inference.huggingface.co/models/deepset/roberta-base-squad2'

# Define the headers with the Bearer token
headers = {
    'Authorization': f'Bearer {bearer_token}',
    'Content-Type': 'application/json',
}


async def extract_answer(question, context):
    response_placeholder = st.empty()
    payload = {
        "inputs": {
            "question": f"{question}",
            "context": f"{context}"
        },
    }
    response = requests.post(url, headers=headers, json=payload)

    return response_placeholder.write(response.json())


def main():
    st.image('https://i.imgur.com/ADPipgb.png', width=200)
    st.header('Question Answering with OCR')
    image = st.file_uploader('Upload an image with text', type=['png', 'jpg'])

    if image is not None:
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        image = Image.open(image)
        extracted_text = ocr.image_to_string(image)
        st.subheader('Extracted Text')
        st.write(extracted_text)

    question_text = st.text_input('What is your question?')
    button = st.button('Submit')

    if button:
        st.caption('You asked: ')
        st.text(question_text)
        with st.spinner("Making Request..."):
            asyncio.run(extract_answer(question_text, extracted_text))


if __name__ == "__main__":
    main()
