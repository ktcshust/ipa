import streamlit as st
import eng_to_ipa as ipa
import requests
import tempfile
import os
import editdistance

API_URL = "https://api-inference.huggingface.co/models/speech31/wav2vec2-large-english-TIMIT-phoneme_v3"
headers = {"Authorization": "Bearer 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"}

def convert_to_ipa(word):
    try:
        ipa_result = ipa.convert(word)
        return ipa_result
    except Exception as e:
        return f"Cannot convert '{word}' to IPA. Error: {e}"

def transcribe_audio(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()["text"]

def calculate_similarity(ipa_text, transcribed_text):
    distance = editdistance.eval(ipa_text, transcribed_text)
    max_length = max(len(ipa_text), len(transcribed_text))
    similarity = 1 - (distance / max_length)
    return similarity

def main():
    st.title("Text and Audio Conversion App")

    # IPA conversion section
    st.header("Convert Text to IPA")
    input_word = st.text_input("Enter an English word:")
    if input_word:
        ipa_result = convert_to_ipa(input_word)
        st.write(f"IPA representation of '{input_word}': {ipa_result}")

    st.markdown("---")

    # Audio transcription section
    st.header("Transcribe Audio to IPA")
    uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, or FLAC)")
    if uploaded_file is not None:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        # Get the path of the temporary file
        temp_file_path = temp_file.name

        # Close and delete the temporary file
        temp_file.close()

        result = transcribe_audio(temp_file_path)

        # Clean up: remove the temporary file
        os.unlink(temp_file_path)

        st.write("Transcribed Text:")
        st.write(result)

        if input_word and uploaded_file:
            similarity_score = calculate_similarity(ipa_result, result)
            st.write(f"Similarity score between IPA and transcribed text: {similarity_score}")

if __name__ == "__main__":
    main()
