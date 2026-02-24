
import os
import asyncio
import io
import traceback
import uuid
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import asyncio
import time
import numpy as np
import soundfile as sf

from indextts.infer_vllm import IndexTTS

tts = None
speaker_lock = asyncio.Lock()


def _get_speaker_json_path() -> str:
    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    return os.path.join(cur_dir, "assets/speaker.json")


def _load_speaker_dict(speaker_path: str) -> dict:
    if not os.path.exists(speaker_path):
        return {}
    with open(speaker_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_speaker_dict(speaker_path: str, speaker_dict: dict) -> None:
    os.makedirs(os.path.dirname(speaker_path), exist_ok=True)
    tmp_path = f"{speaker_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(speaker_dict, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, speaker_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS(model_dir=args.model_dir, gpu_memory_utilization=args.gpu_memory_utilization)

    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    speaker_path = _get_speaker_json_path()
    if os.path.exists(speaker_path):
        speaker_dict = _load_speaker_dict(speaker_path)

        for speaker, audio_paths in speaker_dict.items():
            audio_paths_ = []
            for audio_path in audio_paths:
                audio_paths_.append(os.path.join(cur_dir, audio_path))
            tts.registry_speaker(speaker, audio_paths_)
    yield
    # Clean up the ML models and release the resources
    # ml_models.clear()

app = FastAPI(lifespan=lifespan)

# 添加CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        global tts
        if tts is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "TTS model not initialized"
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "message": "Service is running",
                "timestamp": time.time()
            }
        )
    except Exception as ex:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(ex)
            }
        )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        audio_paths = data["audio_paths"]
        seed = data.get("seed", 8)

        global tts
        sr, wav = await tts.infer(audio_paths, text, seed=seed)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        character = data["character"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )



@app.get("/audio/voices")
async def tts_voices():
    """ additional function to provide the list of available voices, in the form of JSON """
    speaker_path = _get_speaker_json_path()
    if os.path.exists(speaker_path):
        speaker_dict = _load_speaker_dict(speaker_path)
        return speaker_dict
    else:
        return []


@app.post("/audio/clone")
async def audio_clone(
    voice: str = Form(default=""),
    audio: list[UploadFile] = File(...),
):
    """Clone/register speaker voice with one or more reference audio files."""
    try:
        if not audio:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "audio files are required"},
            )

        allowed_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        voice = (voice or "").strip()

        current_file_path = os.path.abspath(__file__)
        cur_dir = os.path.dirname(current_file_path)
        speaker_path = _get_speaker_json_path()

        async with speaker_lock:
            speaker_dict = _load_speaker_dict(speaker_path)

            if not voice:
                voice = f"voice_{uuid.uuid4().hex[:8]}"

            existing_paths = speaker_dict.get(voice, [])
            if not isinstance(existing_paths, list):
                existing_paths = []

            voice_dir = os.path.join(cur_dir, "assets", "speakers", voice)
            os.makedirs(voice_dir, exist_ok=True)

            added_paths = []
            for idx, file in enumerate(audio):
                _, ext = os.path.splitext(file.filename or "")
                ext = ext.lower()
                if ext not in allowed_exts:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "error": f"unsupported audio extension: {ext or '<empty>'}",
                        },
                    )

                audio_bytes = await file.read()
                if not audio_bytes:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "error": f"empty audio file: {file.filename or idx}",
                        },
                    )

                saved_name = f"{int(time.time() * 1000)}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
                abs_audio_path = os.path.join(voice_dir, saved_name)
                with open(abs_audio_path, "wb") as out_f:
                    out_f.write(audio_bytes)

                rel_audio_path = os.path.relpath(abs_audio_path, cur_dir).replace("\\", "/")
                added_paths.append(rel_audio_path)

            updated_paths = existing_paths + added_paths
            speaker_dict[voice] = updated_paths
            _save_speaker_dict(speaker_path, speaker_dict)

            global tts
            abs_audio_paths = [os.path.join(cur_dir, p) for p in updated_paths]
            tts.registry_speaker(voice, abs_audio_paths)

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "voice": voice,
                "added_files": added_paths,
                "total_refs": len(updated_paths),
            },
        )
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )



@app.post("/audio/speech", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_openai(request: Request):
    """ OpenAI competible API, see: https://api.openai.com/v1/audio/speech """
    try:
        data = await request.json()
        text = data["input"]
        character = data["voice"]
        #model param is omitted
        _model = data["model"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="/path/to/IndexTeam/Index-TTS")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    args = parser.parse_args()

    uvicorn.run(app=app, host=args.host, port=args.port)
