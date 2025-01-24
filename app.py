import streamlit as st
import requests
import deepl
from openai import OpenAI
import json
from functools import partial

# ======================================================================
# 1) Page Configuration
# ======================================================================
st.set_page_config(
    page_title="Artikel Übersetzer (nur für intere Tests)",
    page_icon="🌐",
    layout="wide"
)

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
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": """Du bist ein Experte für die Bereinigung von Nachrichtenartikeln. Deine Aufgabe:
1. Entferne alle Navigationselemente, Werbung, Footers, Kommentare und sonstige Website-Elemente
2. Behalte nur den eigentlichen Artikeltext mit Überschrift und Autor
3. Entferne alle Formatierungen und gib nur reinen Text zurück
4. Behalte den journalistischen Inhalt vollständig bei
5. Stelle sicher, dass der Artikel mit dem Titel beginnt
6. Gib den Text ohne zusätzliche Anmerkungen oder Erklärungen zurück"""
                    }
                ]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        ],
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
        messages=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "text",
                        "text": """Erstelle einen verbesserten, aber inhaltlich korrekten Artikel. Der Artikel ist für eine deutsche Nachrichtenseite. Es folgt der Original-Artikel von der Washington Post, der dann mit Deepl Pro automatisch übersetzt wurde. Diese Übersetzungen sind inhaltlich zwar korrekt, aber sprachlich oft nicht gut (zu verschachtelte Sätze, Formulierungen die im Deutschen nicht verwendet werden oder missverständlich sind, Grammatik die eins-zu-eins aus dem Englische übernommen ist und im Deutschen nicht stimmt) 

Übliche Anpassungen, die die Redaktion machen muss sind: 
- Ortsangabe am Beginn des Artikels 
- Langer Satz wurde in 2 Sätze aufgeteilt 
- Ergänzung des Orts zum besseren Verständnis 
- Stilistische Anpassung: Satzkonstruktion wurde verändert 
- Zwischenüberschrift geändert 
- Zwischenüberschrift wurde hinzugefügt 
- Stilistisch nicht gut formuliert, klingt nach maschineller Übersetzung"""
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"# Original Englisch:\n\n{cleaned_text}\n\n# DeepL Übersetzung:\n\n{translated_text}"
                    }
                ]
            }
        ],
        response_format={"type": "text"}
    )
    return response.choices[0].message.content


@st.cache_data(ttl=3600)
def analyze_translation(cleaned_text: str, final_text: str, openai_key: str) -> str:
    """Analyzes the translation quality using GPT-4o-mini with caching"""
    client = OpenAI(api_key=openai_key)
    
    system_prompt = """## Aufgabe
Führe eine systematische Analyse zwischen dem englischen Originaltext und der deutschen Übersetzung durch.
## Prüfkategorien
### 1. FAKTENCHECK
* Vergleiche alle Zahlen, Daten, Maßeinheiten
* Prüfe Namen, Titel, Institutionen auf korrekte Übernahme
* Kontrolliere die präzise Übersetzung aller Zitate
* Stelle sicher, dass Fachbegriffe korrekt übertragen wurden
### 2. VOLLSTÄNDIGKEIT
* Analysiere den Text Absatz für Absatz
* Identifiziere Kernaussagen und wichtige Details im Original
* Prüfe, ob alle relevanten Informationen in der Übersetzung enthalten sind
* Markiere fehlende oder zusätzliche Informationen
### 3. STRUKTUR & KONTEXT
* Vergleiche den logischen Aufbau beider Texte
* Prüfe die korrekte Wiedergabe von Zusammenhängen
* Kontrolliere die angemessene Übertragung des Sprachstils
* Achte auf zielsprachliche Anpassungen
## Ausgabeformat
1. **Abweichungsanalyse**
   * Liste konkrete Unterschiede zwischen Original und Übersetzung
   * Belege mit Textbeispielen
2. **Genauigkeitsbewertung**
   * Bewerte die Präzision der Übersetzung
   * Dokumentiere Stärken und Schwächen
3. **Qualitätseinschätzung**
   * Gib eine fundierte Gesamtbewertung
   * Verweise auf besonders gelungene oder problematische Aspekte
## Methodik
* Gehe systematisch Absatz für Absatz vor
* Belege alle Bewertungen mit konkreten Beispielen
* Achte besonders auf fachliche Korrektheit
* Berücksichtige die Zielgruppe der Übersetzung"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"# Englischer Originaltext:\n\n{cleaned_text}\n\n# Deutsche Übersetzung:\n\n{final_text}"
            }
        ],
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
st.title("🌐 Artikel Übersetzer")

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
        if not st.session_state.processed_text.get('analysis'):
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
        
        if st.session_state.processed_text.get('analysis'):
            st.markdown(st.session_state.processed_text['analysis'])
            
            # Download button for analysis
            st.download_button(
                "Download Prüfbericht",
                st.session_state.processed_text['analysis'],
                file_name="quality_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Download buttons for all versions
    st.write("---")
    st.subheader("Downloads")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    def download_text(text, filename):
        st.download_button(
            f"Download {filename}",
            text,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )
    
    with col1:
        download_text(
            st.session_state.processed_text['original'],
            "original.txt"
        )
    
    with col2:
        download_text(
            st.session_state.processed_text['cleaned'],
            "cleaned.txt"
        )
    
    with col3:
        download_text(
            st.session_state.processed_text['translated'],
            "translated.txt"
        )
    
    with col4:
        download_text(
            st.session_state.processed_text['final'],
            "final.txt"
        )
    
    with col5:
        if st.session_state.processed_text.get('analysis'):
            download_text(
                st.session_state.processed_text['analysis'],
                "analysis.txt"
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
