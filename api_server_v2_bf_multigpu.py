import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7" # 我们将在启动时动态设置，可以删除或注释掉这行

import io
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import argparse
import json
import time
import numpy as np
import soundfile as sf
from pydantic import BaseModel
from typing import Optional, List
import multiprocessing  # <--- 新增导入

# 导入IndexTTS类，请确保路径正确
from indextts.infer_vllm import IndexTTS

# --- 修改部分：在主启动器解析参数，而不是全局 ---
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="要监听的主机地址。")
    parser.add_argument("--port", type=int, default=7860, help="起始端口号，每个GPU将使用一个递增的端口。")
    parser.add_argument("--model_dir", type=str, default="IndexTTS-1.5", help="模型目录的路径。")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5, help="每个进程的GPU内存使用率。")
    # --- 新增参数 ---
    parser.add_argument("--num_gpus", type=int, default=1, help="要使用的GPU数量，将启动相同数量的进程。")
    return parser

# --- 将全局变量和FastAPI应用定义移到函数外部，以便所有进程都能访问 ---
tts = None
args = create_parser().parse_args() # 解析一次以获取模型路径等共享信息

class TTSRequest(BaseModel):
    # api_server.py 原有参数
    text: str
    character: Optional[str] = None
    audio_paths: Optional[List[str]] = None

    # api_v2.py 兼容参数 (多余的参数将被接收但忽略)
    text_lang: Optional[str] = None
    ref_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[List[str]] = None
    prompt_text: Optional[str] = ""
    prompt_lang: Optional[str] = None
    top_k: Optional[int] = 5
    top_p: Optional[float] = 1
    temperature: Optional[float] = 1
    text_split_method: Optional[str] = "cut5"
    batch_size: Optional[int] = 1
    batch_threshold: Optional[float] = 0.75
    split_bucket: Optional[bool] = True
    speed_factor: Optional[float] = 1.0
    fragment_interval: Optional[float] = 0.3
    seed: Optional[int] = -1
    media_type: Optional[str] = "wav"
    streaming_mode: Optional[bool] = False
    parallel_infer: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.35
    sample_steps: Optional[int] = 32
    super_sampling: Optional[bool] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    # 在lifespan中，环境变量CUDA_VISIBLE_DEVICES已经被设置好了
    print(f"PID {os.getpid()}: Initializing TTS model on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}...")
    
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    tts = IndexTTS(model_dir=args.model_dir, cfg_path=cfg_path, gpu_memory_utilization=args.gpu_memory_utilization)

    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    speaker_path = os.path.join(cur_dir, "assets/speaker.json")
    if os.path.exists(speaker_path):
        speaker_dict = json.load(open(speaker_path, 'r'))
        for speaker, audio_paths in speaker_dict.items():
            audio_paths_ = [os.path.join(cur_dir, audio_path) for audio_path in audio_paths]
            tts.registry_speaker(speaker, audio_paths_)
    
    print(f"PID {os.getpid()}: TTS model initialized successfully.")
    yield
    print(f"PID {os.getpid()}: Cleaning up resources.")


# FastAPI应用实例
app = FastAPI(lifespan=lifespan)

# --- 您的API端点（/health, /tts_url, /tts）保持不变 ---
# ... (此处省略您的API端点代码，无需改动)
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
# --- 修改部分：使用新的TTSRequest模型 ---
async def tts_api_url(request: TTSRequest):
    try:
        # --- 修改部分：适配来自 api_v2.py 的参数 ---
        # api_v2.py 使用 ref_audio_path 和 aux_ref_audio_paths
        # 我们将它们组合成 infer 方法需要的列表
        # 同时保留对旧 audio_paths 参数的兼容
        paths_to_use = []
        if request.ref_audio_path:
            paths_to_use.append(request.ref_audio_path)
            if request.aux_ref_audio_paths:
                paths_to_use.extend(request.aux_ref_audio_paths)
        elif request.audio_paths:
            paths_to_use = request.audio_paths
        
        if not paths_to_use:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "No reference audio path provided. Use 'ref_audio_path' or 'audio_paths'."}
            )

        global tts
        # 使用适配后的参数列表和请求中的文本，核心功能不变
        sr, wav = await tts.infer(paths_to_use, request.text)
        
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
# --- 修改部分：使用新的TTSRequest模型 ---
async def tts_api(request: TTSRequest):
    try:
        global tts
        
        # --- 修改部分：增加逻辑判断，以兼容两种调用方式 ---
        # 方式一：如果请求中包含 'character'，则使用原有的、基于预注册角色的推理
        if request.character:
            sr, wav = await tts.infer_with_ref_audio_embed(request.character, request.text)
        
        # 方式二：如果请求中包含 'ref_audio_path'（来自v2前端），则使用基于音频文件的推理
        elif request.ref_audio_path or request.audio_paths:
            paths_to_use = []
            if request.ref_audio_path:
                paths_to_use.append(request.ref_audio_path)
                if request.aux_ref_audio_paths:
                    paths_to_use.extend(request.aux_ref_audio_paths)
            elif request.audio_paths:
                 paths_to_use = request.audio_paths

            # 这种情况下，我们调用 tts.infer，使其行为与 /tts_url 类似
            sr, wav = await tts.infer(paths_to_use, request.text)

        # 如果两种关键参数都没有，则返回错误
        else:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "Request must include either 'character' or 'ref_audio_path'."}
            )

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
# --- 全新的启动器部分 ---
def start_worker(gpu_id: int, host: str, base_port: int, app_module: str):
    """
    启动一个独立的Uvicorn服务进程。
    
    :param gpu_id: 要绑定的GPU的ID。
    :param host: 监听的主机。
    :param base_port: 基础端口号。
    :param app_module: FastAPI应用模块字符串，例如 'your_script_name:app'
    """
    # 关键步骤：为当前进程设置可见的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 计算当前工作进程的端口
    worker_port = base_port + gpu_id
    
    print(f"Starting worker for GPU {gpu_id} on http://{host}:{worker_port}")
    
    # 启动Uvicorn服务
    # 注意：这里的workers参数必须为1，因为我们已经在外部通过multiprocessing来管理进程了。
    uvicorn.run(app_module, host=host, port=worker_port, workers=1)


if __name__ == "__main__":
    # 解析命令行参数
    parser = create_parser()
    args = parser.parse_args()

    if args.num_gpus > 1:
        # --- 多GPU模式 ---
        print(f"Starting in multi-GPU mode with {args.num_gpus} workers.")
        
        # 获取当前脚本的文件名，用于构建app_module字符串
        # 例如 'api_server_v2_bf_multigpu:app'
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        app_module_str = f"{script_name}:app"

        processes = []
        for i in range(args.num_gpus):
            process = multiprocessing.Process(
                target=start_worker,
                args=(i, args.host, args.port, app_module_str)
            )
            processes.append(process)
            process.start()
            
        for process in processes:
            process.join() # 等待所有子进程结束
            
    else:
        # --- 单GPU模式（兼容原有行为）---
        print("Starting in single-GPU mode.")
        # 如果只有一个GPU，直接指定为0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        uvicorn.run(app, host=args.host, port=args.port, workers=1)