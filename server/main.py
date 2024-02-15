from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np
import io
import librosa
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=['*']
)

# Load translation models and tokenizer
tokenizer_en_indic = IndicTransTokenizer(direction="en-indic")
ip_en_indic = IndicProcessor(inference=True)
model_en_indic = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

tokenizer_indic_en = IndicTransTokenizer(direction="indic-en")
ip_indic_en = IndicProcessor(inference=True)
model_indic_en = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True)

class TranslationRequest(BaseModel):
    content: str
    target_language: str

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    content = request.content
    target_language = request.target_language

    # Ensure target_language is in the format languageCode_scriptCode
    if "_" not in target_language:
        raise HTTPException(status_code=400, detail="Invalid target language format. Should be languageCode_scriptCode.")

    if target_language == "English":
        # Translate from Indic to English
        batch = ip_indic_en.preprocess_batch([content], src_lang="hin_Deva", tgt_lang="eng_Latn")
        batch = tokenizer_indic_en(batch, src=True, return_tensors="pt")
        with torch.inference_mode():
            outputs = model_indic_en.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)
        outputs = tokenizer_indic_en.batch_decode(outputs, src=False)
        translated_text = ip_indic_en.postprocess_batch(outputs, lang="eng_Latn")[0]
    else:
        # Translate from English to Indic
        batch = ip_en_indic.preprocess_batch([content], src_lang="eng_Latn", tgt_lang=target_language)
        batch = tokenizer_en_indic(batch, src=True, return_tensors="pt")
        with torch.inference_mode():
            outputs = model_en_indic.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)
        outputs = tokenizer_en_indic.batch_decode(outputs, src=False)
        translated_text = ip_en_indic.postprocess_batch(outputs, lang=target_language)[0]

    return {"translated_text": translated_text}

transcriber_hindi = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec-hindi")
transcriber_bengali = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec_v1_bengali")
transcriber_odia = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec-odia")
transcriber_gujarati = pipeline("automatic-speech-recognition", model="ai4bharat/indicwav2vec_v1_gujarati")

languages = {
    "hindi": transcriber_hindi,
    "bengali": transcriber_bengali,
    "odia": transcriber_odia,
    "gujarati": transcriber_gujarati
}

def resample_to_16k(audio, orig_sr):
    y_resampled = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=16000)
    return y_resampled

def transcribe_and_translate(audio: Tuple[int, np.ndarray], lang: str) -> Tuple[str, float]:
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    y_resampled = resample_to_16k(y, sr)

    if lang not in languages:
        raise HTTPException(status_code=404, detail="Language model not found")
    pipe = languages[lang]
    trans = pipe(y_resampled)
    return trans["text"], trans["text"], trans["text"]

@app.post("/transcribe-translate")
async def transcribe_translate_audio(file: UploadFile = File(...), lang: str = "hindi"):
    content = await file.read()
    audio, sr = librosa.load(io.BytesIO(content), sr=None)
    transcribed_text, translation, _ = transcribe_and_translate((sr, audio), lang)
    return {"transcribed_text": transcribed_text, "translation": translation}

# Run the FastAPI server with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
