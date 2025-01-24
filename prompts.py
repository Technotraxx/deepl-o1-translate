# ======================================================================
# System Prompts
# ======================================================================

CLEANING_SYSTEM_PROMPT = """Du bist ein Experte für die Bereinigung von Nachrichtenartikeln. Deine Aufgabe:
1. Entferne alle Navigationselemente, Werbung, Footers, Kommentare und sonstige Website-Elemente
2. Behalte nur den eigentlichen Artikeltext mit Überschrift und Autor
3. Entferne alle Formatierungen und gib nur reinen Text zurück
4. Behalte den journalistischen Inhalt vollständig bei
5. Stelle sicher, dass der Artikel mit dem Titel beginnt
6. Gib den Text ohne zusätzliche Anmerkungen oder Erklärungen zurück"""

QUALITY_CHECK_SYSTEM_PROMPT = """## Aufgabe
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

# ======================================================================
# Developer Prompts
# ======================================================================

TRANSLATION_DEVELOPER_PROMPT = """Erstelle einen verbesserten, aber inhaltlich korrekten Artikel. Der Artikel ist für eine deutsche Nachrichtenseite. Es folgt der Original-Artikel von der Washington Post, der dann mit Deepl Pro automatisch übersetzt wurde. Diese Übersetzungen sind inhaltlich zwar korrekt, aber sprachlich oft nicht gut (zu verschachtelte Sätze, Formulierungen die im Deutschen nicht verwendet werden oder missverständlich sind, Grammatik die eins-zu-eins aus dem Englische übernommen ist und im Deutschen nicht stimmt) 

Übliche Anpassungen, die die Redaktion machen muss sind: 
- Ortsangabe am Beginn des Artikels 
- Langer Satz wurde in 2 Sätze aufgeteilt 
- Ergänzung des Orts zum besseren Verständnis 
- Stilistische Anpassung: Satzkonstruktion wurde verändert 
- Zwischenüberschrift geändert 
- Zwischenüberschrift wurde hinzugefügt 
- Stilistisch nicht gut formuliert, klingt nach maschineller Übersetzung"""

# ======================================================================
# Message Templates
# ======================================================================

def get_cleaning_messages(text: str) -> list:
    """Returns the messages for the cleaning API call"""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": CLEANING_SYSTEM_PROMPT
                }
            ]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }
    ]

def get_translation_messages(cleaned_text: str, translated_text: str) -> list:
    """Returns the messages for the translation optimization API call"""
    return [
        {
            "role": "developer",
            "content": [
                {
                    "type": "text",
                    "text": TRANSLATION_DEVELOPER_PROMPT
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
    ]

def get_quality_check_messages(cleaned_text: str, final_text: str) -> list:
    """Returns the messages for the quality check API call"""
    return [
        {
            "role": "system",
            "content": QUALITY_CHECK_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"# Englischer Originaltext:\n\n{cleaned_text}\n\n# Deutsche Übersetzung:\n\n{final_text}"
        }
    ]
