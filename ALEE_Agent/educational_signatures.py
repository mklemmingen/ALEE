"""
German Educational DSPy Signatures for Question Generation and Expert Validation
Defines input/output specifications for all DSPy modules
"""

import dspy


# Main Question Generation Signature
class GenerateGermanEducationalQuestions(dspy.Signature):
    """Generiere genau 3 deutsche Bildungsfragen basierend auf Text und Parametern."""
    
    # Input fields - comprehensive SYSARCH parameters
    source_text: str = dspy.InputField(desc="Deutscher Bildungstext als Grundlage")
    
    # Core parameters
    variation: str = dspy.InputField(desc="Schwierigkeitsgrad: leicht/stammaufgabe/schwer")
    question_type: str = dspy.InputField(desc="Fragetyp: multiple-choice/single-choice/true-false/mapping")
    taxonomy_level: str = dspy.InputField(desc="Bloom'sche Taxonomiestufe")
    mathematical_requirement: str = dspy.InputField(desc="Mathematische Anforderung: 0-2")
    
    # Text and reference parameters
    text_reference_explanatory: str = dspy.InputField(desc="Erklärungstext-Referenz")
    
    # Obstacle parameters - comprehensive linguistic challenges
    text_obstacles: str = dspy.InputField(desc="Sprachliche Hindernisse im Grundtext")
    item_obstacles: str = dspy.InputField(desc="Sprachliche Hindernisse in Antwortoptionen")
    instruction_obstacles: str = dspy.InputField(desc="Sprachliche Hindernisse in Anweisungen")
    
    # Instruction parameters
    instruction_explicitness: str = dspy.InputField(desc="Explizitheit der Anweisung: Explizit/Implizit")
    
    # Content parameters  
    irrelevant_information: str = dspy.InputField(desc="Irrelevante Informationen enthalten")
    
    # Output fields - exactly 3 questions with answers
    frage_1: str = dspy.OutputField(desc="Erste deutsche Bildungsfrage")
    frage_2: str = dspy.OutputField(desc="Zweite deutsche Bildungsfrage")
    frage_3: str = dspy.OutputField(desc="Dritte deutsche Bildungsfrage")
    
    antworten_1: str = dspy.OutputField(desc="Antwortoptionen für Frage 1 (kommagetrennt)")
    antworten_2: str = dspy.OutputField(desc="Antwortoptionen für Frage 2 (kommagetrennt)")
    antworten_3: str = dspy.OutputField(desc="Antwortoptionen für Frage 3 (kommagetrennt)")
    
    # Metadata
    generation_rationale: str = dspy.OutputField(desc="Begründung der Fragengenerierung")


# Expert Validation Signatures - German-specific
class ValidateVariationGerman(dspy.Signature):
    """Validiere, ob die Frage dem deutschen Schwierigkeitsgrad entspricht."""
    
    # Input
    frage: str = dspy.InputField(desc="Zu validierende deutsche Frage")
    antworten: str = dspy.InputField(desc="Antwortoptionen")
    ziel_variation: str = dspy.InputField(desc="Ziel-Schwierigkeit: leicht/stammaufgabe/schwer")
    
    # Output - German compliance
    bewertung: int = dspy.OutputField(desc="Bewertung 1-5 (5=perfekte Übereinstimmung)")
    feedback: str = dspy.OutputField(desc="Spezifisches Feedback zur Schwierigkeit")
    vorschlaege: str = dspy.OutputField(desc="Verbesserungsvorschläge wenn Bewertung < 3")
    begruendung: str = dspy.OutputField(desc="Detaillierte Begründung der Bewertung")


class ValidateTaxonomyGerman(dspy.Signature):
    """Validiere Bloom'sche Taxonomiestufe für deutsche Bildungsfragen."""
    
    # Input
    frage: str = dspy.InputField(desc="Deutsche Bildungsfrage")
    antworten: str = dspy.InputField(desc="Antwortoptionen")
    ziel_taxonomie: str = dspy.InputField(desc="Ziel-Taxonomiestufe")
    
    # Output
    bewertung: int = dspy.OutputField(desc="Taxonomie-Bewertung 1-5")
    feedback: str = dspy.OutputField(desc="Taxonomie-Compliance-Analyse")
    vorschlaege: str = dspy.OutputField(desc="Taxonomie-Anpassungsvorschläge")
    begruendung: str = dspy.OutputField(desc="Begründung der Taxonomie-Bewertung")


class ValidateMathematicalGerman(dspy.Signature):
    """Validiere mathematische Komplexität für deutsche Bildungskontext."""
    
    # Input
    frage: str = dspy.InputField(desc="Deutsche Bildungsfrage")
    antworten: str = dspy.InputField(desc="Antwortoptionen")
    ziel_math_stufe: str = dspy.InputField(desc="Ziel-Mathematikstufe 0-2")
    
    # Output
    bewertung: int = dspy.OutputField(desc="Mathematik-Bewertung 1-5")
    feedback: str = dspy.OutputField(desc="Mathematische Komplexitätsanalyse")
    vorschlaege: str = dspy.OutputField(desc="Anpassungen für korrekte Komplexität")
    begruendung: str = dspy.OutputField(desc="Begründung der Mathematik-Bewertung")


class ValidateObstacleGerman(dspy.Signature):
    """Validiere sprachliche Hindernisse in deutschen Bildungsfragen."""
    
    # Input
    frage: str = dspy.InputField(desc="Deutsche Bildungsfrage")
    antworten: str = dspy.InputField(desc="Antwortoptionen")
    ziel_hindernisse: str = dspy.InputField(desc="Erwartete sprachliche Hindernisse")
    
    # Output
    bewertung: int = dspy.OutputField(desc="Hindernis-Bewertung 1-5")
    feedback: str = dspy.OutputField(desc="Sprachliche Hindernisanalyse")
    vorschlaege: str = dspy.OutputField(desc="Vorschläge zur Hindernisanpassung")
    begruendung: str = dspy.OutputField(desc="Begründung der Hindernis-Bewertung")


class ValidateInstructionGerman(dspy.Signature):
    """Validiere Anweisungsklarheit und Explizitheit für deutsche Bildung."""
    
    # Input
    frage: str = dspy.InputField(desc="Deutsche Bildungsfrage")
    antworten: str = dspy.InputField(desc="Antwortoptionen")
    ziel_explizitheit: str = dspy.InputField(desc="Ziel-Explizitheit: Explizit/Implizit")
    
    # Output
    bewertung: int = dspy.OutputField(desc="Anweisungs-Bewertung 1-5")
    feedback: str = dspy.OutputField(desc="Anweisungsklarheitsanalyse")
    vorschlaege: str = dspy.OutputField(desc="Verbesserungen der Anweisungsklarheit")
    begruendung: str = dspy.OutputField(desc="Begründung der Anweisungs-Bewertung")


class ValidateContentGerman(dspy.Signature):
    """Validiere Inhaltsrelevanz und irrelevante Informationen."""
    
    # Input
    frage: str = dspy.InputField(desc="Deutsche Bildungsfrage")
    antworten: str = dspy.InputField(desc="Antwortoptionen")
    ziel_relevanz: str = dspy.InputField(desc="Ziel-Inhaltsrelevanz")
    
    # Output
    bewertung: int = dspy.OutputField(desc="Inhalts-Bewertung 1-5")
    feedback: str = dspy.OutputField(desc="Inhaltsrelevanzanalyse")
    vorschlaege: str = dspy.OutputField(desc="Inhaltsverbesserungsvorschläge")
    begruendung: str = dspy.OutputField(desc="Begründung der Inhalts-Bewertung")


# Consensus and Decision Signatures
class ExpertConsensusGerman(dspy.Signature):
    """Bestimme Konsens aus allen deutschen Expertenbewertungen."""
    
    # Input - aggregated expert data
    experten_bewertungen: str = dspy.InputField(desc="Liste aller Expertenbewertungen")
    experten_feedback: str = dspy.InputField(desc="Kombiniertes Expertenfeedback")
    experten_vorschlaege: str = dspy.InputField(desc="Alle Verbesserungsvorschläge")
    
    # Output - intelligent consensus  
    genehmigt: str = dspy.OutputField(desc="'ja' wenn alle Bewertungen >= 3, sonst 'nein'")
    verbesserungspriorität: str = dspy.OutputField(desc="Wichtigste benötigte Verbesserung")
    konsens_begründung: str = dspy.OutputField(desc="Begründung der Konsens-Entscheidung")
    zusammengefasste_vorschlaege: str = dspy.OutputField(desc="Intelligente Synthese aller Expertvorschläge")


class QuestionImprovementGerman(dspy.Signature):
    """Verbessere deutsche Bildungsfragen basierend auf Expertenvorschlägen."""
    
    # Input
    original_frage: str = dspy.InputField(desc="Ursprüngliche deutsche Frage")
    original_antworten: str = dspy.InputField(desc="Ursprüngliche Antwortoptionen")
    verbesserungsvorschlaege: str = dspy.InputField(desc="Expertvorschläge zur Verbesserung")
    priorität: str = dspy.InputField(desc="Hauptverbesserungspriorität")
    
    # Output
    verbesserte_frage: str = dspy.OutputField(desc="Verbesserte deutsche Bildungsfrage")
    verbesserte_antworten: str = dspy.OutputField(desc="Verbesserte Antwortoptionen")
    verbesserungs_begründung: str = dspy.OutputField(desc="Begründung der Verbesserungen")


# CSV Export Signature
class GenerateSysArchCSV(dspy.Signature):
    """Generiere SYSARCH-konforme CSV-Daten aus validierten deutschen Fragen."""
    
    # Input
    validierte_fragen: str = dspy.InputField(desc="Alle 3 validierten deutschen Fragen")
    parameter_daten: str = dspy.InputField(desc="Alle SYSARCH-Parameter")
    c_id: str = dspy.InputField(desc="Fragen-ID")
    
    # Output - complete CSV row
    csv_zeile: str = dspy.OutputField(desc="Komplette SYSARCH-CSV-Zeile mit allen Spalten")
    spalten_mapping: str = dspy.OutputField(desc="Mapping der Parameter zu CSV-Spalten")


# Quality Assessment Signature
class AssessQuestionQuality(dspy.Signature):
    """Bewerte die Gesamtqualität generierter deutscher Bildungsfragen."""
    
    # Input
    fragen: str = dspy.InputField(desc="Alle 3 generierten Fragen")
    expertenbewertungen: str = dspy.InputField(desc="Zusammenfassung aller Expertenbewertungen")
    
    # Output
    gesamtqualität: int = dspy.OutputField(desc="Gesamtqualitätsbewertung 1-10")
    stärken: str = dspy.OutputField(desc="Identifizierte Stärken der Fragen")
    schwächen: str = dspy.OutputField(desc="Identifizierte Schwächen der Fragen")
    empfehlungen: str = dspy.OutputField(desc="Empfehlungen für zukünftige Verbesserungen")