import streamlit as st
import requests
import deepl
from openai import OpenAI
import json
from functools import partial
from datetime import datetime

from prompts import (
    get_cleaning_messages,
    get_translation_messages,
    get_quality_check_messages
)

# ======================================================================
# 0) Page Configuration
# ======================================================================
st.set_page_config(
    page_title="Artikel Übersetzer (nur für intere Tests)",
    page_icon="🌐",
    layout="wide"
)

# ======================================================================
# 1) Helper Functions
# ======================================================================
def get_file_prefix(text: str) -> str:
    """Extracts first 5 words from the text and adds current date"""
    try:
        # Get first line (title)
        first_line = text.split('\n')[0] if text else "Untitled"
        # Get first 5 words
        words = first_line.split()[:5]
        title_prefix = '_'.join(words)
        # Clean title (remove special chars)
        title_prefix = ''.join(c if c.isalnum() or c == '_' else '' for c in title_prefix)
        # Add date
        date_str = datetime.now().strftime("%Y_%m_%d")
        return f"{title_prefix}_{date_str}"
    except Exception:
        # Fallback if something goes wrong
        return f"article_{datetime.now().strftime('%Y_%m_%d')}"

# ======================================================================
# 2) Session State Initialization
# ======================================================================
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = {
        'original': '',
        'cleaned': '',
        'translated': '',
        'final': '',
        'analysis': ''  # New field for the analysis report
    }

# ======================================================================
# 3) Sidebar for API Keys
# ======================================================================
with st.sidebar:
    st.title("API Konfiguration")
    
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Geben Sie Ihren OpenAI API-Schlüssel ein",
        key="openai_key"
    )
    
    jina_key = st.text_input(
        "Jina API Key",
        type="password",
        help="Geben Sie Ihren Jina API-Schlüssel ein",
        key="jina_key"
    )
    
    deepl_key = st.text_input(
        "DeepL API Key",
        type="password",
        help="Geben Sie Ihren DeepL API-Schlüssel ein",
        key="deepl_key"
    )

# ======================================================================
# 4) Cached Functions
# ======================================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def extract_text_from_url(url: str, jina_key: str) -> str:
    """Extracts text from a URL using Jina AI Reader with caching"""
    jina_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {jina_key}',
        'X-Return-Format': 'text'
    }
    
    response = requests.get(jina_url, headers=headers)
    response.raise_for_status()
    return response.text

@st.cache_data(ttl=3600)
def clean_text_with_gpt(text: str, openai_key: str) -> str:
    """Cleans the text using GPT-4o-mini with caching"""
    client = OpenAI(api_key=openai_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=get_cleaning_messages(text),
        response_format={"type": "text"},
        temperature=0,
        max_completion_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

@st.cache_data(ttl=3600)
def translate_text(text: str, deepl_key: str) -> str:
    """Translates text from English to German using DeepL with caching"""
    translator = deepl.Translator(deepl_key)
    result = translator.translate_text(
        text,
        source_lang="EN",
        target_lang="DE",
        formality="more"
    )
    return result.text

@st.cache_data(ttl=3600)
def optimize_translation(cleaned_text: str, translated_text: str, openai_key: str) -> str:
    """Optimizes the translation using OpenAI with caching"""
    client = OpenAI(api_key=openai_key)
    
    response = client.chat.completions.create(
        model="o1",
        messages=get_translation_messages(cleaned_text, translated_text),
        response_format={"type": "text"}
    )
    return response.choices[0].message.content


@st.cache_data(ttl=3600)
def analyze_translation(cleaned_text: str, final_text: str, openai_key: str) -> str:
    """Analyzes the translation quality using GPT-4o-mini with caching"""
    client = OpenAI(api_key=openai_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=get_quality_check_messages(cleaned_text, final_text),
        response_format={"type": "text"},
        temperature=0,
        max_completion_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

# ======================================================================
# 5) Main App Layout
# ======================================================================
st.title("🌐 Artikel Übersetzer (Test-Version nur für internen Gebrauch")

# URL Input
url = st.text_input(
    "Artikel URL",
    placeholder="https://www.example.com/article",
    help="Fügen Sie die URL des zu übersetzenden Artikels ein"
)

# Process Button
if st.button("Artikel verarbeiten", type="primary", use_container_width=True):
    if not all([url, openai_key, jina_key, deepl_key]):
        st.error("Bitte füllen Sie alle erforderlichen Felder aus (URL und API-Keys)")
    else:
        try:
            # Reset all stored texts including analysis
            st.session_state.processed_text = {
                'original': '',
                'cleaned': '',
                'translated': '',
                'final': '',
                'analysis': ''
            }
            
            with st.status("Verarbeite Artikel...", expanded=True) as status:
                # Extract text
                st.write("🔍 Extrahiere Text von URL...")
                raw_text = extract_text_from_url(url, jina_key)
                st.session_state.processed_text['original'] = raw_text
                
                # Clean text
                st.write("🧹 Bereinige Text...")
                cleaned_text = clean_text_with_gpt(raw_text, openai_key)
                st.session_state.processed_text['cleaned'] = cleaned_text
                
                # Translate text
                st.write("🔄 Übersetze Text...")
                translated_text = translate_text(cleaned_text, deepl_key)
                st.session_state.processed_text['translated'] = translated_text
                
                # Optimize translation
                st.write("✨ Optimiere Übersetzung...")
                final_text = optimize_translation(cleaned_text, translated_text, openai_key)
                st.session_state.processed_text['final'] = final_text
                
                status.update(label="Verarbeitung abgeschlossen! ✅", state="complete")

        except Exception as e:
            st.error(f"Fehler bei der Verarbeitung: {str(e)}")

# Results Display
if st.session_state.processed_text['original']:
    st.write("---")
    
    # Create tabs for different versions including new Analysis tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Original", 
        "🧹 Bereinigt", 
        "🔄 DeepL Übersetzung",
        "✨ Finale Version",
        "🔍 Qualitätsprüfung"
    ])
    
    with tab1:
        st.text_area(
            "Original Text",
            st.session_state.processed_text['original'],
            height=400,
            disabled=True
        )
        
    with tab2:
        st.text_area(
            "Bereinigter Text",
            st.session_state.processed_text['cleaned'],
            height=400,
            disabled=True
        )
        
    with tab3:
        st.text_area(
            "DeepL Übersetzung",
            st.session_state.processed_text['translated'],
            height=400,
            disabled=True
        )
        
    with tab4:
        st.text_area(
            "Finale Version",
            st.session_state.processed_text['final'],
            height=400,
            disabled=True
        )

    with tab5:
        # Always show the analysis button
        if st.button("Qualitätsprüfung durchführen", type="primary", use_container_width=True):
            try:
                with st.status("Führe Qualitätsprüfung durch...", expanded=True) as status:
                    analysis_result = analyze_translation(
                        st.session_state.processed_text['cleaned'],
                        st.session_state.processed_text['final'],
                        openai_key
                    )
                    st.session_state.processed_text['analysis'] = analysis_result
                    status.update(label="Qualitätsprüfung abgeschlossen! ✅", state="complete")
            except Exception as e:
                st.error(f"Fehler bei der Qualitätsprüfung: {str(e)}")
        
        # Show analysis results if available
        if st.session_state.processed_text.get('analysis'):
            st.markdown("## Prüfbericht")
            st.markdown(st.session_state.processed_text['analysis'])

    # Download buttons for all versions
    st.write("---")
    st.subheader("Downloads")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    def download_text(text, button_label, filename):
        st.download_button(
            f"Download {button_label}",
            text,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )
    
    # Get file prefix from cleaned text (which contains the title)
    file_prefix = get_file_prefix(st.session_state.processed_text['cleaned'])
    
    with col1:
        download_text(
            st.session_state.processed_text['original'],
            "Original",
            f"{file_prefix}_original.txt"
        )
    
    with col2:
        download_text(
            st.session_state.processed_text['cleaned'],
            "Bereinigt",
            f"{file_prefix}_bereinigt.txt"
        )
    
    with col3:
        download_text(
            st.session_state.processed_text['translated'],
            "DeepL",
            f"{file_prefix}_deepl.txt"
        )
    
    with col4:
        download_text(
            st.session_state.processed_text['final'],
            "Final",
            f"{file_prefix}_final.txt"
        )
    
    with col5:
        if st.session_state.processed_text.get('analysis'):
            download_text(
                st.session_state.processed_text['analysis'],
                "Prüfbericht",
                f"{file_prefix}_pruefbericht.txt"
            )

    
# ======================================================================
# 6) Footer
# ======================================================================
st.write("---")
st.markdown("""
<div style='text-align: center'>
    <p>Entwickelt mit ❤️ | Powered by OpenAI, DeepL & Jina AI</p>
</div>
""", unsafe_allow_html=True)
