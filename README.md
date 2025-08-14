# Speech to text

## Overview
This project is designed to convert user's voice to text, as speech-to-text (ASR).

## Architectural Decisions

This project uses **OpenAI Whisper** as the core speech recognition engine to convert audio input to text. Whisper is chosen for its:

- **High accuracy** in speech-to-text conversion across multiple languages
- **Robust performance** with various audio qualities and accents
- **Efficient processing** suitable for real-time applications
- **Open-source availability** allowing for cost-effective deployment

The service architecture follows a RESTful API design pattern, providing a simple HTTP endpoint for audio file upload and transcription processing.

## Running the Services

### Prerequisites:
Ensure you have Python 3.8+ and the required dependencies installed. 

ASR Service:
This service runs on port 8011.
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8011
```
