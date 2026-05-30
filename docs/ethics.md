# Ethische Überlegungen: Skin Lesion Classifier

**ZHAW AI-Applications Abschlussprojekt**  
**Datum:** 2026-05-30  
**Autor:** Jeff0559

---

## 1. Kein Ersatz für medizinische Diagnose

Dieses System dient **ausschliesslich Bildungszwecken** und ist kein zugelassenes Medizinprodukt.

- Das Modell wurde auf einem öffentlichen Forschungsdatensatz (HAM10000) trainiert
- Die Genauigkeit ist nicht ausreichend für klinische Entscheidungen
- **Bei Hautveränderungen immer einen Dermatologen aufsuchen**
- Kein Ergebnis dieses Tools darf als medizinische Empfehlung interpretiert werden

**Regulatorische Einschränkung:** In der EU gelten KI-basierte Medizinprodukte als Klasse III (höchstes Risiko) unter der MDR (Medical Device Regulation). Dieses Tool erfüllt diese Anforderungen nicht und darf nicht klinisch eingesetzt werden.

---

## 2. Bias im Dataset

### 2.1 Demographischer Bias

Der HAM10000 Datensatz weist bekannte Verzerrungen auf:

| Bias-Typ | Beschreibung | Auswirkung |
|----------|-------------|-----------|
| Hautfarbe | Majoritär helle Hautfarben | Schlechtere Performance bei dunklerer Haut |
| Alter | Überrepräsentation 40-70 Jahre | Weniger zuverlässig bei Kindern/Älteren |
| Geographie | Europäisch-australischer Fokus | Andere Populationen unterrepräsentiert |
| Geschlecht | Geringe Imbalance | Leicht reduzierte Genauigkeit bei Frauen |

### 2.2 Klinischer Bias

- **Confirmation Bias:** Bilder wurden von Dermatologen gelabelt – bereits klinisch bekannte Läsionen sind überrepräsentiert
- **Equipment Bias:** Dermatoskopie-Bilder, nicht Smartphone-Bilder – andere Bildqualität kann Genauigkeit reduzieren
- **Selection Bias:** Patienten, die Dermatologen aufsuchen, nicht repräsentativ für Allgemeinbevölkerung

### 2.3 Massnahmen

- Klassengewichtung im Training zur Kompensation von Klassenimbalance
- Klare Dokumentation der Limitierungen im UI
- Modell-Performance nach demographischen Gruppen separat evaluieren (empfohlen für Produktionssysteme)

---

## 3. Verantwortungsvoller Einsatz

### Was dieses Tool KANN:
- Bildungsinteressen zu Hautläsionen wecken
- Forschung zu multimodaler KI demonstrieren
- Bewusstsein für Selbstuntersuchung fördern

### Was dieses Tool NICHT KANN:
- Klinische Diagnosen stellen
- Biopsien oder Behandlungen empfehlen
- Als alleinige Grundlage für medizinische Entscheidungen dienen
- Alle Hauttypen gleich zuverlässig analysieren

### Empfohlene Safeguards für Produktionssysteme:
1. Obligatorischer Disclaimer bei jedem Ergebnis
2. "Not a doctor" Hinweis in der UI
3. Logging und Auditierung aller Anfragen
4. Rate Limiting um Missbrauch zu verhindern
5. Keine Speicherung von Patientenbildern ohne Einwilligung

---

## 4. Datenschutz bei Hautbildern

### Sensibilität
Hautbilder sind besonders sensible personenbezogene Daten:
- Sie können Rückschlüsse auf Gesundheitszustand, Alter, ethnische Zugehörigkeit ermöglichen
- Sie fallen unter besondere Schutzkategorien der DSGVO (Art. 9)
- Gesundheitsdaten genießen erhöhten Schutz in der Schweiz (revDSG)

### Implementierte Schutzmassnahmen
- Keine persistente Speicherung von Upload-Bildern im Demo-System
- Keine Weitergabe von Bildern an Drittparteien
- API-Calls an Anthropic enthalten **keine** Bilder, nur extrahierte Textfeatures
- `.env` Datei mit API-Keys wird nie committed (`.gitignore` gesetzt)
- `data/raw/` wird nie committed

### DSGVO/revDSG Konformität
Für einen Produktionseinsatz wären folgende Massnahmen notwendig:
- Privacy Impact Assessment (PIA)
- Einwilligungserklärung für Bildverarbeitung
- Recht auf Löschung implementieren
- Datenverarbeitungsvertrag mit Cloud-Anbietern

---

## 5. Transparenz und Explainability

Das System bietet mehrere Erklärbarkeitsebenen:

| Methode | Block | Beschreibung |
|---------|-------|-------------|
| Grad-CAM | CV | Visualisiert welche Bildregionen die ResNet50-Entscheidung beeinflussten |
| SHAP Values | ML | Zeigt welche Features (CV+NLP+Metadata) die finale Vorhersage prägen |
| Claude Erklärung | NLP | Generiert verständliche Erklärung der Ergebnisse in Plaintext |
| Konfidenzwerte | Alle | Alle Vorhersagen mit Wahrscheinlichkeiten |

**Wichtig:** Explainability macht das System nicht diagnostisch valide. Es hilft nur, die Funktionsweise zu verstehen.

---

## 6. Regulatorischer Kontext

| Regulierung | Relevanz | Status |
|-------------|----------|--------|
| EU AI Act (2024) | Hochrisiko-KI (medizinisch) | Nicht konform – nur Bildungszwecke |
| EU MDR | Medizinprodukt Klasse III | Nicht zugelassen |
| DSGVO | Gesundheitsdaten | Demo-Modus: keine persistente Speicherung |
| revDSG (CH) | Schweizer Datenschutz | Demo-Modus: konform |

---

## 7. Fazit

Dieses Projekt demonstriert die technischen Möglichkeiten multimodaler KI in der Dermatologie. Die ethischen Herausforderungen – insbesondere Bias, Datenschutz und das Risiko falscher Sicherheit bei Patienten – unterstreichen, warum solche Systeme strenger Regulierung, klinischer Validierung und transparenter Kommunikation bedürfen, bevor sie in der Praxis eingesetzt werden dürfen.

**Kernbotschaft:** Dieses Tool zeigt, was KI kann. Ein Dermatologe zeigt, was du brauchst.

---

*Dieses Dokument wurde im Rahmen des ZHAW AI-Applications Kurses erstellt.*
