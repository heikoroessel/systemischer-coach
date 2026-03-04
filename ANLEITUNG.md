# 🎙 Systemischer Coach – Installationsanleitung

Schritt-für-Schritt-Anleitung, um deinen Coach live zu schalten.
Dauer: ca. 30–45 Minuten.

---

## Was du brauchst

- Einen Computer mit Browser (Chrome oder Firefox empfohlen)
- Deine OpenAI API Keys
- Ca. 30 Minuten Zeit

---

## Schritt 1: Accounts anlegen

### 1a) GitHub-Account
GitHub ist eine Plattform, auf der du deinen Code speicherst. Railway holt ihn von dort automatisch.

1. Gehe zu **https://github.com** und klicke auf „Sign up"
2. E-Mail-Adresse, Passwort und Benutzername eingeben
3. Konto bestätigen (E-Mail)
4. Fertig – du hast ein GitHub-Konto

### 1b) Railway-Account
Railway ist dein Server in der Cloud.

1. Gehe zu **https://railway.app**
2. Klicke auf „Login" → „Login with GitHub"
3. GitHub-Konto verknüpfen → bestätigen
4. Fertig – du bist bei Railway angemeldet

### 1c) Pinecone-Account
Pinecone ist deine Vektordatenbank für die Podcast-Inhalte.

1. Gehe zu **https://www.pinecone.io**
2. Klicke auf „Sign Up Free"
3. Mit Google oder E-Mail registrieren
4. Nach dem Login: Klicke oben rechts auf deinen Namen → **„API Keys"**
5. Kopiere deinen API-Key (er sieht aus wie: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)
6. Notiere ihn – du brauchst ihn in Schritt 3

---

## Schritt 2: Code auf GitHub hochladen

Du musst den Code-Ordner (den du von mir bekommen hast) auf GitHub hochladen.

### Den Code-Ordner finden
- Du hast einen ZIP-Ordner namens **`systemischer-coach`** heruntergeladen
- Entpacke ihn auf deinem Computer (Rechtsklick → „Alle extrahieren")

### GitHub Repository erstellen
1. Gehe zu **https://github.com/new**
2. Repository name: `systemischer-coach`
3. Wähle „Private" (damit nicht jeder deinen Code sieht)
4. Klicke auf „Create repository"
5. Du siehst jetzt eine leere Repository-Seite

### Code hochladen (Upload-Methode – kein Terminal nötig!)
1. Auf der leeren Repository-Seite: Klicke auf **„uploading an existing file"**
2. Ziehe den **gesamten entpackten Ordner-Inhalt** in das Upload-Fenster
   - NICHT den Ordner selbst, sondern die **Dateien darin**:
     - `main.py`
     - `requirements.txt`
     - `Procfile`
     - `railway.json`
     - Ordner `static/` (mit `index.html` und `admin.html`)
3. Klicke auf **„Commit changes"** → **„Commit changes"** bestätigen
4. ✅ Dein Code ist jetzt auf GitHub

---

## Schritt 3: Railway einrichten

### Neues Projekt erstellen
1. Gehe zu **https://railway.app/dashboard**
2. Klicke auf **„New Project"**
3. Wähle **„Deploy from GitHub repo"**
4. Wähle dein Repository `systemischer-coach` aus
5. Railway erkennt den Code automatisch und startet den Build

### Umgebungsvariablen eintragen (API Keys)
Das sind die geheimen Schlüssel, damit dein Coach mit OpenAI und Pinecone kommunizieren kann.

1. Klicke in Railway auf dein Projekt
2. Klicke auf **„Variables"** (linke Seite oder oben)
3. Klicke auf **„Add Variable"** und trage folgende Werte ein:

| Variable | Wert |
|---|---|
| `OPENAI_API_KEY` | Dein OpenAI API Key (beginnt mit `sk-...`) |
| `PINECONE_API_KEY` | Dein Pinecone API Key |
| `PINECONE_INDEX` | `systemischer-coach` |
| `ADMIN_PASSWORD` | Ein Passwort deiner Wahl (z.B. `MeinCoach2024!`) |

> **Wo finde ich meinen OpenAI API Key?**
> → Gehe zu **https://platform.openai.com/api-keys**
> → Klicke auf „Create new secret key"
> → Kopiere ihn sofort (er wird nur einmal angezeigt!)

4. Nach dem Eintragen aller Variablen: Railway startet automatisch neu

### Deine URL finden
1. Klicke in Railway auf **„Settings"**
2. Dort findest du unter **„Domains"** eine URL wie:
   `systemischer-coach-production.up.railway.app`
3. Klicke auf **„Generate Domain"** wenn noch keine vorhanden
4. ✅ Das ist deine App-Adresse!

---

## Schritt 4: App testen

### Coach-Interface
1. Öffne deine Railway-URL im Browser
2. Du siehst den grünen Kreis
3. Klicke auf „Gespräch beginnen"
4. Browser fragt nach Mikrofon-Erlaubnis → **„Zulassen"** klicken
5. Warte kurz – der Coach begrüßt dich auf Deutsch
6. ✅ Wenn du den Coach hörst, funktioniert alles!

> **Hinweis:** Der Coach funktioniert auch ohne Podcast-Daten in der Datenbank.
> Er nutzt dann sein allgemeines systemisches Coaching-Wissen.
> Mit deinen Podcast-Daten wird er aber viel spezifischer!

### Admin-Interface
1. Öffne: `deine-url.railway.app/admin`
2. Gib dein Admin-Passwort ein
3. Du siehst die Datenbank-Übersicht

---

## Schritt 5: Podcast-Episoden einpflegen

So bringst du deinen Coach zum Lernen!

1. Öffne das Admin-Interface: `/admin`
2. Passwort eingeben
3. Episoden-Titel eingeben (z.B. `Episode 001 – Systemische Grundlagen`)
4. MP3-Datei per Drag & Drop hochladen
5. Auf **„Jetzt verarbeiten"** klicken
6. Abwarten (je nach Länge der Episode: 1–5 Minuten):
   - ✅ MP3 hochgeladen
   - ✅ Whisper transkribiert (das dauert am längsten)
   - ✅ Text aufgeteilt
   - ✅ Embeddings erstellt
   - ✅ In Datenbank gespeichert
7. Nächste Episode hochladen
8. Wiederhole für alle Episoden

> **Tipp:** Du kannst Episoden in beliebiger Reihenfolge hochladen.
> Es gibt kein Maximum – alle 300+ Episoden passen problemlos rein.

---

## Kosten im Überblick

| Dienst | Kosten |
|---|---|
| **Railway** | Starter Plan: 5 $/Monat (reicht für den Anfang) |
| **OpenAI Whisper** | ca. 0,006 $ pro Minute Audio (300 Episoden à 60 Min ≈ 108 $) |
| **OpenAI Embeddings** | sehr günstig, ca. 1–2 $ für alle Episoden |
| **OpenAI Realtime API** | ca. 0,06 $ pro Gesprächsminute |
| **Pinecone** | Starter (Free Tier) reicht für den Anfang |

> **Empfehlung:** Lade erst 5–10 Episoden hoch und teste, ob alles funktioniert,
> bevor du alle 300 hochlädst.

---

## Häufige Probleme

### „Mikrofon funktioniert nicht"
→ Browser-Einstellungen prüfen: Klicke auf das Schloss-Symbol in der Adressleiste → Mikrofon erlauben

### „Verbindungsfehler beim Gespräch"
→ OpenAI API Key prüfen: Hat der Key noch Guthaben? → https://platform.openai.com/usage

### „Upload schlägt fehl"
→ MP3-Datei darf max. 25 MB groß sein (OpenAI Whisper-Limit)
→ Längere Episoden mit einem Tool wie Audacity in Teile aufteilen

### Railway Build schlägt fehl
→ Prüfe ob alle Dateien hochgeladen wurden (besonders `requirements.txt`)
→ In Railway: Klicke auf „Deployments" → klicke auf den fehlgeschlagenen Build → Logs lesen

---

## Fertig! 🎉

Dein systemischer Coach ist live. Du erreichst ihn unter:
- **Coach:** `https://deine-url.railway.app`
- **Admin:** `https://deine-url.railway.app/admin`

Bei Fragen: Screenshot machen und mir zeigen – ich helfe weiter!
