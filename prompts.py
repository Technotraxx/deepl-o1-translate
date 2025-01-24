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

TRANSLATION_DEVELOPER_PROMPT = """Erstelle einen verbesserten, aber inhaltlich korrekten Artikel. Der Artikel ist für eine deutsche Nachrichtenseite. Es folgt der Original-Artikel von der Washington Post, der dann mit Deepl Pro automatisch übersetzt wurde. Diese Übersetzungen sind inhaltlich meistens zwar korrekt, aber sprachlich oft nicht gut (zu verschachtelte Sätze, Formulierungen die im Deutschen nicht verwendet werden oder missverständlich sind, Grammatik die eins-zu-eins aus dem Englische übernommen ist und im Deutschen nicht stimmt, teilweise werden Zitate nicht sauber übersetzt und zu stark verkürzt.) Gehe Satz- und Absatzweise vor und stelle sicher, dass wirklich alle Elemente des Originals in der finalen Übersetzung vorhanden sind. 

Achte darauf, dass die Präzision der Übersetzung nicht zu Lasten der Lesbarkeit geht. Kürze und verständliche Sätze sind wichtig. Nur bei direkten Zitaten (üblicherweise durch Anführungszeichen erkenntlich) sollten im Ganzen übersetzt werden und nicht aufgeteilt werden.
Die englischen Originale verwenden oft viele Kommas, Neben- und Schachtelsätze oder andere komplizierte Strukturen. Daher sollte die finale Version diese Strukturen aufbrechen und für deutsche Leser besser aufbereiten - ohne die Aussagen und Fakten zu verfälschen.

Der erste Satz im Artikel darf kein Schachtelsatz (Hauptsatz mit meheren Nebensätzen). Dieser muss so umgeschrieben werden, dass ein einfacher und verständlicher Einstieg in den Artikel möglich ist.
Die Fakten und Aussagen aus dem Original-Artikel müssen aber erhalten bleiben - es ist aber akzeptabel wenn das dann auf zwei oder sogar drei Sätze aufgeteilt wird.

Übliche Anpassungen, die die Redaktion machen muss sind: 
- Zitate müssen korrigiert werden, weil sie bei der automatischen Übersetzung nicht korrekt übertragen wurden
- fehlende wörtliche Zitate und präzise Formulierungen führen zu einem Verlust an Tiefe und Authentizität
- Die Umformulierung von zentralen Aussagen kann die Wahrnehmung des Themas beeinflussen.
- Ortsangabe am Beginn des Artikels 
- Langer Satz wurde in 2 Sätze aufgeteilt 
- Ergänzung des Orts zum besseren Verständnis 
- Stilistische Anpassung: Satzkonstruktion wurde verändert 
- Zwischenüberschrift geändert 
- Zwischenüberschrift wurde hinzugefügt 
- Stilistisch nicht gut formuliert, klingt nach maschineller Übersetzung

Wichtig: Ein Analyse-System wird am Ende den Original-Artikel sowie die finale von Dir erstelle Überabeitung überprüfen. Stelle sicher, dass das Analyse-System keine Fehler und Abweichungen findet!

*Ausgabe-Format:*
Überschrift (Headline)

Ortsmarke
Artikel-Text (Body) mit Zwischenüberschriften

Nach jeder Zwischenüberschrift eine Zeile Abstand einfügen
"""

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
