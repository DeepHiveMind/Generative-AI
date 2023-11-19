import streamlit as st
import PyPDF2
import requests
import cohere
import json
from gradio_client import Client

co = cohere.Client('H447nwcOSEFLLXkyvv78aFsWx0qqOl6SD2PXsDMO')

# PDF reader function
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

# Function to query Hugging Face models
def query_api(api_url, payload):
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(api_url, headers=headers, json=payload)
    return response

# Streamlit app
def main():
    st.title("PDF Text Processing App")
    
    # Upload a PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if pdf_file is not None:
        # Read the PDF file
        pdf_text = read_pdf(pdf_file.name)
        st.subheader("How you want to Process Text from PDF")

        # Text summarization
        if st.checkbox("Generate Summary"):
            response = co.summarize(text = pdf_text,length='long',format='auto',model='summarize-xlarge',
            additional_command='hi',
            temperature=1 )
            pdf_summary=response.summary
            st.subheader("Summary")
            st.write(response.summary)
            
            # Translation
            if st.checkbox("Translate to Another Language"):
                options = st.selectbox("Select the type of data source",options=['T2TT (Text to Text translation)','T2ST (Text to Speech translation)'])
                options2 = st.selectbox("Select the source Lang",options=['Afrikaans', 'Amharic', 'Armenian', 'Assamese', 'Basque', 'Belarusian', 'Bengali', 'Bosnian', 'Bulgarian', 'Burmese', 'Cantonese', 'Catalan', 'Cebuano', 'Central Kurdish', 'Croatian', 'Czech', 'Danish', 'Dutch', 'Egyptian Arabic', 'English', 'Estonian', 'Finnish', 'French', 'Galician', 'Ganda', 'Georgian', 'German', 'Greek', 'Gujarati', 'Halh Mongolian', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Korean', 'Kyrgyz', 'Lao', 'Lithuanian', 'Luo', 'Macedonian', 'Maithili', 'Malayalam', 'Maltese', 'Mandarin Chinese', 'Marathi', 'Meitei', 'Modern Standard Arabic', 'Moroccan Arabic', 'Nepali', 'North Azerbaijani', 'Northern Uzbek', 'Norwegian Bokmål', 'Norwegian Nynorsk', 'Nyanja', 'Odia', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Serbian', 'Shona', 'Sindhi', 'Slovak', 'Slovenian', 'Somali', 'Southern Pashto', 'Spanish', 'Standard Latvian', 'Standard Malay', 'Swahili', 'Swedish', 'Tagalog', 'Tajik', 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Vietnamese', 'Welsh', 'West Central Oromo', 'Western Persian', 'Yoruba', 'Zulu'])
                options3 = st.selectbox("Select the output Lang",options=['Afrikaans', 'Amharic', 'Armenian', 'Assamese', 'Basque', 'Belarusian', 'Bengali', 'Bosnian', 'Bulgarian', 'Burmese', 'Cantonese', 'Catalan', 'Cebuano', 'Central Kurdish', 'Croatian', 'Czech', 'Danish', 'Dutch', 'Egyptian Arabic', 'English', 'Estonian', 'Finnish', 'French', 'Galician', 'Ganda', 'Georgian', 'German', 'Greek', 'Gujarati', 'Halh Mongolian', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Korean', 'Kyrgyz', 'Lao', 'Lithuanian', 'Luo', 'Macedonian', 'Maithili', 'Malayalam', 'Maltese', 'Mandarin Chinese', 'Marathi', 'Meitei', 'Modern Standard Arabic', 'Moroccan Arabic', 'Nepali', 'North Azerbaijani', 'Northern Uzbek', 'Norwegian Bokmål', 'Norwegian Nynorsk', 'Nyanja', 'Odia', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Serbian', 'Shona', 'Sindhi', 'Slovak', 'Slovenian', 'Somali', 'Southern Pashto', 'Spanish', 'Standard Latvian', 'Standard Malay', 'Swahili', 'Swedish', 'Tagalog', 'Tajik', 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Vietnamese', 'Welsh', 'West Central Oromo', 'Western Persian', 'Yoruba', 'Zulu'])
                if st.button("Translate"):
                    client = Client("https://facebook-seamless-m4t.hf.space/")
                    result = client.predict(
                        options,
                        "file",	# str in 'Audio source' Radio component
                        "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",	# str (filepath or URL to file) in 'Input speech' Audio component
                        "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",	# str (filepath or URL to file)	in 'Input speech' Audio component
                        pdf_summary,	# str in 'Input text' Textbox component
                        options2,	    # str (Option from: 
                        options3,	    # str (Option from: ['Bengali', 'Catalan', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 'Finnish', 'French', 'German', 'Hindi', 'Indonesian', 'Italian', 'Japanese', 'Korean', 'Maltese', 'Mandarin Chinese', 'Modern Standard Arabic', 'Northern Uzbek', 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Slovak', 'Spanish', 'Swahili', 'Swedish', 'Tagalog', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Vietnamese', 'Welsh', 'Western Persian']) in 'Target language' Dropdown component
                        api_name="/run"
                    )
                    st.subheader("Translated Text")
                    st.write(result[1])
                    if (options=='T2ST (Text to Speech translation)'):
                        st.audio(result[0], format="audio/wav")
                        

        # NER
        if st.checkbox("Named Entity Recognition"):
            NER_API_URL = "https://api-inference.huggingface.co/models/2rtl3/mn-xlm-roberta-base-named-entity"
            pdf_text = read_pdf(pdf_file.name)
            output = query_api(NER_API_URL,{
                "inputs": pdf_text
                # "wait_for_model": True
            })
            st.subheader("NER")
            selected_info = [{"entity_group": entry["entity_group"], "score": entry["score"], "word": entry["word"]} for entry in output.json()]
            ner=json.loads(json.dumps(selected_info))
            st.write(ner)

if __name__ == "__main__":
    main()