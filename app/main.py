from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ASR Service", description="Automatic Speech Recognition Service")

# Lazy load model to avoid startup errors
asr_model = None

def get_asr_model():
    """
    Get ASR model instance (lazy loading)
    """
    global asr_model
    if asr_model is None:
        try:
            from app.models.whisper_asr import WhisperASR
            logger.info("Initializing ASR model...")
            asr_model = WhisperASR()
            logger.info("ASR model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ASR model: {e}")
            raise HTTPException(status_code=503, detail=f"ASR service unavailable: {str(e)}")
    return asr_model

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Try to get model status
        model = get_asr_model()
        return {
            "status": "healthy", 
            "service": "asr-service",
            "model_loaded": model is not None,
            "device": getattr(model, 'device', 'unknown')
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "asr-service", 
            "error": str(e)
        }

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...), text: Optional[str] = Form(None)):
    """
    Audio transcription endpoint
    
    Args:
        file: Audio file
        text: Optional reference text, if provided returns this text, otherwise returns transcription result
    
    Returns:
        Transcription result or reference text
    """
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Validate file type
    if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
        logger.warning(f"Invalid file type: {audio_file.content_type}")
    
    temp_path = f"temp_{audio_file.filename}"
    
    try:
        # Save temporary file
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file")
            f.write(content)
        
        # If reference text is provided and not empty, return the reference text
        if text and text.strip():
            logger.info(f"Using provided reference text: {text.strip()}")
            return {
                "transcription": text.strip(),
                "source": "reference_text",
                "language": "unknown",
                "confidence": 1.0,
                "message": "Using provided reference text"
            }
        else:
            # Get ASR model
            model = get_asr_model()
            
            # Perform speech recognition
            logger.info(f"Transcribing audio file: {audio_file.filename}")
            result = model.transcribe(temp_path)
            
            logger.info(f"Transcription result: {result.get('transcription', 'No transcription')}")
            
            # Check for errors
            if "error" in result:
                raise HTTPException(status_code=500, detail=f"Transcription error: {result['error']}")
            
            # Ensure consistent return format
            if isinstance(result, dict):
                return {
                    "transcription": result.get("transcription", ""),
                    "source": "asr_model",
                    "language": result.get("language", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "segments": result.get("segments", []),
                    "processing_info": result.get("processing_info", {})
                }
            else:
                return {
                    "transcription": str(result),
                    "source": "asr_model",
                    "language": "unknown",
                    "confidence": 0.0
                }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio file {audio_file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")