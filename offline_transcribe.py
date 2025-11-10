import os
import io
import re
import sys
import wave
import time
import argparse
import tempfile
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio

# ---------- WebRTC VAD ----------
try:
    import webrtcvad

    _WEBRTCVAD_OK = True
except Exception:
    webrtcvad = None
    _WEBRTCVAD_OK = False

# ---------- ASR 后端：Whisper / SenseVoice ----------
from openai import OpenAI  # DeepSeek 兼容 OpenAI SDK

# Whisper 相关
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as hf_pipeline
)

# SenseVoice (funasr) 相关
try:
    from funasr import AutoModel

    _FUNASR_OK = True
except ImportError:
    AutoModel = None
    _FUNASR_OK = False

# 声纹
from speechbrain.inference.classifiers import EncoderClassifier


# =========================== DeepSeek 提示词 ===========================
def build_emr_prompt(raw_dialogue: str) -> List[dict]:
    system_msg = (
        "你是一名临床文书与中文电子病历（EMR）规范化专家。\n"
        "请基于提供的口述就诊对话（含ASR转写），严格按顺序完成以下两项任务：\n\n"
        "**任务一：对话规范化**\n"
        "1.  **纠正**：识别并纠正 ASR 误识别、口语化表述、以及不当的断句。\n"
        "2.  **重述**：使用专业的医学术语与标准单位重述（例如：日期/时间、药物剂量、身体部位等）。\n"
        "3.  **输出**：在第一个标题下，输出经过上述处理后的**完整对话全文**。\n"
        "4.  **保留角色**：**必须** 严格保留原始对话中的说话人标注（例如 'User A:', 'User B:'），**禁止** 将其替换为任何其他角色。\n\n"
        "**任务二：EMR结构化提取**\n"
        "1.  **提取**：基于**规范化后**的对话内容，提取信息并逐项填写病历。\n"
        "2.  **约束**：\n"
        "    - 严格保留事实与时序，不进行主观推断或臆造信息。\n"
        "    - 对于对话中确实**未提及**的信息，在相应项目中明确写入“**未提及**”。\n\n"
        "**输出格式**（中文 Markdown）：\n"
        "## 规范化ASR转写全文\n"
        "...\n\n"
        "## 主诉\n"
        "...\n\n"
        "## 现病史\n"
        "...\n\n"
        "## 既往史\n"
        "...\n\n"
        "## 过敏史\n"
        "...\n\n"
        "## 用药史\n"
        "...\n\n"
        "## 家族史\n"
        "...\n\n"
        "## 初步印象 / 鉴别诊断\n"
        "...\n\n"
        "## 建议与计划\n"
        "..."
    )
    user_msg = (
        "以下是逐句 ASR 转写的就诊口述（包含说话人标注）。"
        "请据此生成**规范化全文**和**结构化中文 EMR**：\n\n"
        f"{raw_dialogue}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def call_deepseek(raw_dialogue: str, temperature: float = 0.2) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY 环境变量。请先 export DEEPSEEK_API_KEY=...")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = build_emr_prompt(raw_dialogue)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def split_deepseek_outputs(full_text: str) -> Tuple[str, str]:
    norm_pattern = r"##\s*规范化ASR转写全文\s*(.*?)(?=\n##\s*[^#]+|\Z)"
    m = re.search(norm_pattern, full_text, flags=re.S | re.M)
    if m:
        normalized = m.group(1).strip()
        start = m.start()
        end = m.end()
        emr_rest = (full_text[:start] + full_text[end:]).strip()
        emr_rest = re.sub(r"\n{3,}", "\n\n", emr_rest).strip()
        return normalized, emr_rest if emr_rest else full_text
    else:
        return "", full_text.strip()


# =========================== 工具函数 ===========================
def pcm16_energy(frame: np.ndarray) -> float:
    x = frame.astype(np.float32) / 32768.0
    return float(np.mean(np.abs(x)))


def trim_silence(pcm: np.ndarray, sr: int, thr: float = 0.002, win_ms: int = 20) -> np.ndarray:
    """简单前后端静音裁剪（能量阈值）。"""
    win = int(sr * win_ms / 1000)
    if win <= 0:
        return pcm
    n = (pcm.size // win) * win
    x = pcm[:n].reshape(-1, win)
    e = np.mean(np.abs(x.astype(np.float32) / 32768.0), axis=1)
    idx = np.where(e > thr)[0]
    if idx.size == 0:
        return pcm
    s, t = idx[0] * win, (idx[-1] + 1) * win
    return pcm[s:t]


def peak_normalize(pcm: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """峰值归一化到 ±peak*32767，提升弱音可辨性。"""
    a = np.max(np.abs(pcm))
    if a < 1:
        return pcm
    scale = int(32767 * peak) / float(a)
    y = (pcm.astype(np.float32) * scale).clip(-32768, 32767).astype(np.int16)
    return y


def wav_bytes_from_pcm(pcm: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.astype(np.int16, copy=False).tobytes())
    return buf.getvalue()


# =========================== HybridVAD（WebRTC + 能量） ===========================
class HybridVAD:
    """
    20ms 帧；进入句子前预垫 120ms；静音累计 >= silence_limit_s 结束一句；
    可选 webrtcvad 融合能量门限，更稳健；支持最短/最长句长控制；结束时裁剪+归一化。
    """

    def __init__(
            self,
            sr_out: int = 16000,
            frame_ms: int = 20,
            # ==================== [ 解决重叠 ] ====================
            prepad_ms: int = 80,  # 原为 120, 缩短预缓冲，减少抓取上一句句尾
            silence_limit_s: float = 0.6,
            min_utt_s: float = 0.25,
            max_utt_s: float = 25.0,
            # ==================== [ 解决幻觉 ] ====================
            energy_min: float = 0.008,  # 原为 0.003, 提高最小能量门限
            energy_mul: float = 3.5,  # 原为 2.5, 提高噪声容忍乘数
            noise_alpha: float = 0.95,
            use_webrtc: bool = True,
            webrtc_aggr: int = 2,  # 0-3，越大越激进
            # ==================== [ 解决幻觉 ] ====================
            trim_thr: float = 0.008,  # 原为 0.002, 提高裁剪阈值，配合能量门限
            tail_keep_ms: int = 0,
    ):
        self.sr = sr_out
        self.frame_len = int(sr_out * frame_ms / 1000)
        self.prepad_len = int(sr_out * prepad_ms / 1000)
        self.silence_limit_s = silence_limit_s
        self.min_utt_s = min_utt_s
        self.max_utt_s = max_utt_s
        self.energy_min = energy_min
        self.energy_mul = energy_mul
        self.noise_alpha = noise_alpha
        self.trim_thr = trim_thr
        self.tail_keep = int(sr_out * tail_keep_ms / 1000)

        self.use_webrtc = use_webrtc and _WEBRTCVAD_OK
        if self.use_webrtc:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(int(np.clip(webrtc_aggr, 0, 3)))

        self.reset()

    def reset(self):
        self.noise_ema = 0.0
        self._carry = np.zeros(0, dtype=np.int16)
        self._pre_buffer = np.zeros(0, dtype=np.int16)
        self._utter_active = False
        self._utter_pcm = np.zeros(0, dtype=np.int16)
        self._sil_accum = 0.0
        self._dur_accum = 0.0

    def _energy_vad(self, frame: np.ndarray) -> bool:
        e = pcm16_energy(frame)
        if self.noise_ema == 0.0:
            self.noise_ema = e
        thr = max(self.energy_min, self.noise_ema * self.energy_mul)
        speech = (e >= thr)
        if not speech:
            self.noise_ema = self.noise_alpha * self.noise_ema + (1 - self.noise_alpha) * e
        return speech

    def _webrtc_vad(self, frame: np.ndarray) -> bool:
        if not self.use_webrtc:
            return False
        # 16k/16bit/mono/20ms 要求
        return self.vad.is_speech(frame.tobytes(), sample_rate=self.sr)

    def feed(self, pcm_chunk: np.ndarray) -> List[bytes]:
        out = []
        cat = pcm_chunk if self._carry.size == 0 else np.concatenate([self._carry, pcm_chunk], axis=0)
        n = (cat.size // self.frame_len) * self.frame_len
        frames = cat[:n].reshape(-1, self.frame_len)
        self._carry = cat[n:]

        for fr in frames:
            sp = self._webrtc_vad(fr) or self._energy_vad(fr)

            if not self._utter_active:
                # 维护预垫
                if self._pre_buffer.size == 0:
                    self._pre_buffer = fr.copy()
                else:
                    self._pre_buffer = np.concatenate([self._pre_buffer, fr], axis=0)[-self.prepad_len:]
                if sp:
                    self._utter_active = True
                    self._utter_pcm = self._pre_buffer.copy()
                    self._utter_pcm = np.concatenate([self._utter_pcm, fr], axis=0)
                    self._sil_accum = 0.0
                    self._dur_accum = self._utter_pcm.size / float(self.sr)
            else:
                self._utter_pcm = np.concatenate([self._utter_pcm, fr], axis=0)
                self._dur_accum = self._utter_pcm.size / float(self.sr)

                if sp:
                    self._sil_accum = 0.0
                else:
                    self._sil_accum += self.frame_len / float(self.sr)

                # 触发结束条件：静音到阈值或句子过长
                if self._sil_accum >= self.silence_limit_s or self._dur_accum >= self.max_utt_s:
                    pcm = self._utter_pcm
                    if self.tail_keep > 0:
                        # 保留尾部一点点静音，通常不需要
                        pass

                    pcm = trim_silence(pcm, self.sr, thr=self.trim_thr)
                    dur = max(1, pcm.size) / float(self.sr)
                    if dur >= self.min_utt_s:
                        pcm = peak_normalize(pcm, peak=0.9)
                        out.append(wav_bytes_from_pcm(pcm, self.sr))

                    # 重置一句
                    self._utter_active = False
                    self._utter_pcm = np.zeros(0, dtype=np.int16)
                    self._sil_accum = 0.0
                    self._pre_buffer = np.zeros(0, dtype=np.int16)
                    self._dur_accum = 0.0

        return out

    def flush(self) -> List[bytes]:
        out = []
        if self._utter_active:
            pcm = self._utter_pcm
            if self._carry.size > 0:
                pcm = np.concatenate([pcm, self._carry], axis=0)
            pcm = trim_silence(pcm, self.sr, thr=self.trim_thr)
            dur = max(1, pcm.size) / float(self.sr)
            if dur >= self.min_utt_s:
                pcm = peak_normalize(pcm, peak=0.9)
                out.append(wav_bytes_from_pcm(pcm, self.sr))
        self.reset()
        return out


# =========================== ASR 后端实现 ===========================
class WhisperASR:
    def __init__(self, device: Optional[str] = None, dtype: Optional[torch.dtype] = None,
                 model_id: str = "openai/whisper-large-v3"):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.model_id = model_id

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(self.device)
            processor = AutoProcessor.from_pretrained(model_id)
            self.pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.dtype,
                device=self.device,
            )
            self.ok = True
        except Exception as e:
            print(f"[ERROR] Whisper 初始化失败。请检查 transformers/torch/CUDA/模型。原因：{e}")
            self.ok = False

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        if not self.ok:
            return ""
        with io.BytesIO(wav_bytes) as f:
            wav, sr = torchaudio.load(f)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        arr = wav.squeeze(0).cpu().numpy().astype(np.float32)

        # 定义传递给 Whisper 的生成参数，以强力抑制幻觉
        generate_kwargs = {
            # 1. 强制中文转写：杜绝 "All right", "еще" 等外语幻觉
            "language": "chinese",
            "task": "transcribe",

            # 2. 启用置信度 VAD：
            #    如果模型对转写结果的平均置信度 (logprob) 低于 -1.0，则判为静音
            "logprob_threshold": -1.0,

            # 3. 启用内置 VAD：
            #    如果模型认为 "无语音" 的概率高于 0.6，则判为静音
            "no_speech_threshold": 0.6,

            # 显式提供 temperature。当使用 logprob_threshold 时，
            # 内部逻辑需要一个非 None 的 temperature 值。
            "temperature": 0.0
        }

        # 增加 batch_size=1 来抑制 transformers pipeline 的 GPU 警告
        res = self.pipe(
            {"array": arr, "sampling_rate": 16000},
            batch_size=1,  # 保持 batch_size=1
            generate_kwargs=generate_kwargs  # 传入上述反幻觉参数
        )

        text = res.get("text", "") or ""
        text = re.sub(r'<\|[^>]+\|>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# =========================== SenseVoice ASR 实现 ===========================
class SenseVoiceASR:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_id = "iic/SenseVoiceSmall"

        if not _FUNASR_OK:
            print("[ERROR] SenseVoice ASR 需要 funasr 库。请先执行: pip install funasr")
            self.ok = False
            return

        try:
            # 加载模型
            self.model = AutoModel(
                model=self.model_id,
                device=self.device,
                disable_update=True,
                disable_download=False
            )
            self.ok = True
        except Exception as e:
            print(f"[ERROR] SenseVoice (funasr) 初始化失败。请检查 funasr 安装。原因：{e}")
            self.ok = False

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        """
        使用 funasr.AutoModel 进行转写。
        注意：funasr 的 generate 似乎偏好文件路径输入，因此我们使用临时文件。
        """
        if not self.ok:
            return ""

        tmp_file_path = None
        try:
            # 1. 创建一个临时 WAV 文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(wav_bytes)
                tmp_file_path = tmp_file.name

            # 2. 调用 funasr.AutoModel.generate
            result = self.model.generate(
                input=tmp_file_path,
                batch_size_s=0,
                language="auto"  # 自动检测（SenseVoice 支持多语言）
            )

            # 3. 解析结果 (参考 voice_doctor_webui.py)
            if result and len(result) > 0:
                raw_text = result[0].get("text", "")
                # 清理 <|...|> 等特殊标记
                clean_text = re.sub(r'<\|[^>]+\|>', '', raw_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                return clean_text

            return ""

        except Exception as e:
            print(f"[ASR-SenseVoice] 错误: {e}")
            return ""

        finally:
            # 4. 确保删除临时文件
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    print(f"[WARN] 删除临时文件 {tmp_file_path} 失败: {e}")


# =========================== 声纹识别 ===========================
def load_speaker_model() -> Optional[EncoderClassifier]:
    save_dir = os.path.join("/tmp", "speechbrain_models", "spkrec-ecapa-voxceleb")
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=save_dir,
        run_opts={"device": device}
    )
    model.eval()
    return model


def wav_bytes_to_tensor_16k_mono(wav_bytes: bytes) -> Optional[torch.Tensor]:
    try:
        with io.BytesIO(wav_bytes) as f:
            wav, sr = torchaudio.load(f)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        return wav
    except Exception:
        return None


def recognize_speaker(
        wav_bytes: bytes,
        spk_model: Optional[EncoderClassifier],
        known_speakers: List[Dict],
        spk_threshold: float = 0.35,
        ema_alpha: float = 0.15,
) -> Tuple[str, List[Dict]]:
    if spk_model is None:
        return "User", known_speakers

    signal = wav_bytes_to_tensor_16k_mono(wav_bytes)
    if signal is None:
        return "User", known_speakers

    dur_s = signal.shape[1] / 16000.0
    if dur_s < 0.35:  # 短句声纹不稳定，避免误判
        return "User", known_speakers

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    signal = signal.to(device)

    with torch.no_grad():
        emb = spk_model.encode_batch(signal).squeeze().cpu()
        emb = torch.nn.functional.normalize(emb, p=2, dim=0)

    if not known_speakers:
        return "User A", [{"name": "User A", "embedding": emb, "num_samples": 1}]

    sim_fn = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    best, best_sim = None, -1.0
    for sp in known_speakers:
        sim = sim_fn(emb, sp["embedding"]).item()
        if sim > best_sim:
            best_sim = sim
            best = sp

    if best and best_sim >= spk_threshold:
        updated = []
        for sp in known_speakers:
            if sp is best:
                upd = (1 - ema_alpha) * sp["embedding"] + ema_alpha * emb
                upd = torch.nn.functional.normalize(upd, p=2, dim=0)
                updated.append({"name": sp["name"], "embedding": upd, "num_samples": sp["num_samples"] + 1})
            else:
                updated.append(sp)
        return best["name"], updated
    else:
        idx = len(known_speakers)
        if idx < 26:
            name = f"User {chr(ord('A') + idx)}"
        else:
            name = f"User {chr(ord('A') + (idx % 26))}{idx // 26}"
        known_speakers = known_speakers + [{"name": name, "embedding": emb, "num_samples": 1}]
        return name, known_speakers


# =========================== 音频读取 ===========================
def load_audio_to_int16_mono_16k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.ndim > 1 and wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = wav.squeeze(0).numpy().astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    wav_i16 = (wav * 32768.0).astype(np.int16)
    return wav_i16


def audio_duration_seconds(path: str) -> float:
    info = torchaudio.info(path)
    return float(info.num_frames) / float(info.sample_rate)


# =========================== 主流程 ===========================
def main():
    parser = argparse.ArgumentParser(description="离线流式ASR + 声纹 + DeepSeek校正/病历（HybridVAD版）")
    parser.add_argument("--audio", default="./wav/demo3.wav", help="输入音频路径（建议 ≤ 40 分钟）")
    parser.add_argument("--output_dir", default=".", help="输出目录（保存 asr.txt, history.txt）")
    parser.add_argument("--chunk_sec", type=float, default=0.3, help="馈入 VAD 的块时长（秒）")

    # ==================== [ 增加 ASR 后端选择 ] ====================
    parser.add_argument(
        "--asr_backend",
        type=str,
        default="sensevoice",
        choices=["whisper", "sensevoice"],
        help="选择ASR后端：'whisper' (large-v3) 或 'sensevoice' (iic/SenseVoiceSmall)。"
    )
    # ===================================================================

    # VAD 切分参数
    parser.add_argument("--silence", type=float, default=0.4, help="静音累计阈值（秒），达到则结束一句")
    parser.add_argument("--min-utt", type=float, default=0.25, help="导出的最短句长（秒）")
    parser.add_argument("--max-utt", type=float, default=15.0, help="导出的最长句长（秒），防止超长不切")

    parser.add_argument("--spk-threshold", type=float, default=0.4, help="声纹匹配阈值（余弦相似度）")

    parser.add_argument("--use-deepseek", action="store_true", default=True,
                        help="[开关] 是否调用 DeepSeek API 进行校正与病历生成。")

    args = parser.parse_args()

    audio_path = args.audio
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(audio_path):
        print(f"[ERROR] 找不到音频文件: {audio_path}")
        sys.exit(1)

    dur = audio_duration_seconds(audio_path)
    if dur > 40 * 60:
        print(f"[WARN] 音频时长 {dur / 60:.1f} 分钟，超过建议的 40 分钟。继续处理。")

    # ==================== [ 根据参数动态加载 ASR ] ====================
    asr = None
    if args.asr_backend == "whisper":
        print("[INFO] 加载 ASR 后端：Whisper (openai/whisper-large-v3)...")
        asr = WhisperASR()
    elif args.asr_backend == "sensevoice":
        print("[INFO] 加载 ASR 后端：SenseVoice (iic/SenseVoiceSmall)...")
        asr = SenseVoiceASR()

    if asr is None or not asr.ok:
        print(f"[ERROR] {args.asr_backend} ASR 初始化失败，无法继续。")
        sys.exit(1)
    # ===================================================================

    print("[INFO] 加载声纹模型 speechbrain/spkrec-ecapa-voxceleb ...")
    spk_model = load_speaker_model()

    print("[INFO] 读取音频并重采样到 16k 单声道 int16 ...")
    wav_i16 = load_audio_to_int16_mono_16k(audio_path)

    #  显示 ASR 后端信息
    print(f"[INFO] ASR 后端：{asr.model_id}")  # 两个类都必须有 model_id 属性

    if _WEBRTCVAD_OK:
        print("[INFO] 分句：WebRTC VAD + 能量门限（Hybrid）")
    else:
        print("[WARN] 未安装 webrtcvad，仅使用能量门限分句。建议：pip install webrtcvad")

    # 初始化分句器
    vad = HybridVAD(
        sr_out=16000,
        silence_limit_s=args.silence,
        min_utt_s=args.min_utt,
        max_utt_s=args.max_utt,
        use_webrtc=True,
        webrtc_aggr=3,  # 保持激进的 WebRTC 模式
    )

    # 逐块馈入，模拟“流式”
    hop = int(16000 * args.chunk_sec)
    known_speakers: List[Dict] = []
    lines: List[str] = []
    total_segments = 0

    print("[INFO] 开始流式分句 + 逐句ASR（终端将实时显示每句结果）...\n")
    t0 = time.time()
    for s in range(0, len(wav_i16), hop):
        seg = wav_i16[s: s + hop]
        for wav_bytes in vad.feed(seg):
            spk_name, known_speakers = recognize_speaker(
                wav_bytes, spk_model, known_speakers, spk_threshold=args.spk_threshold
            )
            # ==================== [ ASR调用 ] ====================
            # 无论选择哪个后端，都调用 .transcribe_wav_bytes() 接口
            text = asr.transcribe_wav_bytes(wav_bytes)
            # ===================================================================

            # 若仍为空，给一次极简退路：放松正则清理已做，这里只作为占位不丢句
            if not text:
                text = "[未识别]"
            line = f"{spk_name}: {text}"
            print(line, flush=True)
            lines.append(line)
            total_segments += 1

    # 冲刷尾句
    for wav_bytes in vad.flush():
        spk_name, known_speakers = recognize_speaker(
            wav_bytes, spk_model, known_speakers, spk_threshold=args.spk_threshold
        )
        text = asr.transcribe_wav_bytes(wav_bytes)
        if not text:
            text = "[未识别]"
        line = f"{spk_name}: {text}"
        print(line, flush=True)
        lines.append(line)
        total_segments += 1

    used = time.time() - t0
    print(f"\n[INFO] 完成本地 ASR：共 {total_segments} 句，用时 {used:.1f}s")

    raw_dialogue = "\n".join(lines).strip()
    if not raw_dialogue:
        print("[WARN] 未得到有效句子文本，停止。")
        sys.exit(0)

    asr_out = os.path.join(out_dir, "asr.txt")

    if not args.use_deepseek:
        print("\n[INFO] --use-deepseek 未开启，跳过 DeepSeek 校正与病历生成。")
        with open(asr_out, "w", encoding="utf-8") as f:
            f.write(raw_dialogue)
        print(f"[OK] 已保存原始/占位 ASR 转写：{asr_out}")
        print("[DONE] (未执行 DeepSeek)")
        sys.exit(0)

    # --- DeepSeek 二次转译 + 生成 EMR ---
    print("\n[INFO] 调用 DeepSeek 进行校正与病历生成 ...")
    try:
        full_text = call_deepseek(raw_dialogue, temperature=0.2)
    except Exception as e:
        print(f"[ERROR] DeepSeek 调用失败: {e}")
        print("[HINT] 请确认已设置 DEEPSEEK_API_KEY，或稍后重试。")
        sys.exit(2)

    normalized, emr_text = split_deepseek_outputs(full_text)
    hist_out = os.path.join(out_dir, "history.txt")

    with open(asr_out, "w", encoding="utf-8") as f:
        f.write((normalized or "").strip())
    with open(hist_out, "w", encoding="utf-8") as f:
        f.write((emr_text or full_text or "").strip())

    print(f"[OK] 已保存规范化转写：{asr_out}")
    print(f"[OK] 已保存结构化病历：{hist_out}")
    print("[DONE]")


if __name__ == "__main__":
    main()
