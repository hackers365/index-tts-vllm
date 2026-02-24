import os
import asyncio
import io
import traceback
import uuid
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import soundfile as sf

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

tts = None
speaker_lock = asyncio.Lock()


def _get_speaker_json_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "speaker.json")


def _load_speaker_dict() -> dict:
    speaker_path = _get_speaker_json_path()
    if not os.path.exists(speaker_path):
        return {}
    with open(speaker_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_speaker_dict(speaker_dict: dict) -> None:
    speaker_path = _get_speaker_json_path()
    os.makedirs(os.path.dirname(speaker_path), exist_ok=True)
    tmp_path = f"{speaker_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(speaker_dict, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, speaker_path)


def _resolve_voice_to_audio_path(voice: str) -> str | None:
    if not voice:
        return None
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(voice) and os.path.exists(voice):
        return voice
    rel_path = os.path.join(cur_dir, voice)
    if os.path.exists(rel_path):
        return rel_path
    speaker_dict = _load_speaker_dict()
    refs = speaker_dict.get(voice, [])
    if isinstance(refs, list) and refs:
        return os.path.join(cur_dir, refs[0])
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    if tts is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "message": "TTS model not initialized"})
    return JSONResponse(status_code=200, content={"status": "healthy", "message": "Service is running", "timestamp": time.time()})


@app.get("/audio/voices")
async def tts_voices():
    return _load_speaker_dict()


@app.post("/audio/clone")
async def audio_clone(voice: str = Form(default=""), audio: list[UploadFile] = File(...)):
    try:
        if not audio:
            return JSONResponse(status_code=400, content={"status": "error", "error": "audio files are required"})
        allowed_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        voice = (voice or "").strip()
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        async with speaker_lock:
            speaker_dict = _load_speaker_dict()
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
                    return JSONResponse(status_code=400, content={"status": "error", "error": f"unsupported audio extension: {ext or '<empty>'}"})

                audio_bytes = await file.read()
                if not audio_bytes:
                    return JSONResponse(status_code=400, content={"status": "error", "error": f"empty audio file: {file.filename or idx}"})

                saved_name = f"{int(time.time() * 1000)}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
                abs_path = os.path.join(voice_dir, saved_name)
                with open(abs_path, "wb") as out_f:
                    out_f.write(audio_bytes)
                added_paths.append(os.path.relpath(abs_path, cur_dir).replace("\\", "/"))

            updated_paths = existing_paths + added_paths
            speaker_dict[voice] = updated_paths
            _save_speaker_dict(speaker_dict)

        return JSONResponse(status_code=200, content={"status": "ok", "voice": voice, "added_files": added_paths, "total_refs": len(updated_paths)})
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(status_code=500, content={"status": "error", "error": str(tb_str)})


@app.post("/tts_url", responses={200: {"content": {"application/octet-stream": {}}}, 500: {"content": {"application/json": {}}}})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(status_code=500, content={"status": "error", "error": "情感向量之和不能超过1.5，请调整后重试。"})
        else:
            vec = None

        sr, wav = await tts.infer(
            spk_audio_prompt=spk_audio_path,
            text=text,
            output_path=None,
            emo_audio_prompt=emo_ref_path,
            emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_control_method == 3),
            emo_text=emo_text,
            use_random=emo_random,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
        )

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(status_code=500, content={"status": "error", "error": str(tb_str)})


@app.post("/audio/speech", responses={200: {"content": {"application/octet-stream": {}}}, 500: {"content": {"application/json": {}}}})
async def tts_api_openai(request: Request):
    try:
        data = await request.json()
        text = data["input"]
        voice = data["voice"]
        _model = data.get("model")

        spk_audio_path = _resolve_voice_to_audio_path(voice)
        if spk_audio_path is None:
            return JSONResponse(status_code=400, content={"status": "error", "error": f"voice not found: {voice}"})

        sr, wav = await tts.infer(
            spk_audio_prompt=spk_audio_path,
            text=text,
            output_path=None,
            emo_audio_prompt=None,
            emo_alpha=1.0,
            emo_vector=None,
            use_emo_text=False,
            emo_text=None,
            use_random=False,
        )

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(status_code=500, content={"status": "error", "error": str(tb_str)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)
