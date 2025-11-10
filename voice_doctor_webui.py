import io
import wave
import numpy as np
import torch
import streamlit as st
import tempfile
import os
import re
import av
import time
from typing import Optional, List, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    AudioProcessorBase
)

from funasr import AutoModel
from MeloTTS.melo.api import TTS
from av.audio.resampler import AudioResampler

# ======================= 声纹识别导入 =======================
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

# ======================= DeepSeek API 工具 =======================
from openai import OpenAI

# ======================= WebRTC VAD =======================
import webrtcvad


@st.cache_resource
def init_deepseek_client() -> Optional["OpenAI"]:
    """初始化 DeepSeek API 工具（从环境变量读取 DEEPSEEK_API_KEY）"""
    if OpenAI is None:
        st.warning("未检测到 openai 库，请先执行：pip install openai")
        return None
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        st.warning("未设置 DEEPSEEK_API_KEY 环境变量，无法调用 DeepSeek。")
        return None
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        return client
    except Exception as e:
        st.error(f"初始化 DeepSeek API 工具失败：{e}")
        return None


def _build_emr_prompt(raw_dialogue: str) -> List[dict]:
    """
    构造 DeepSeek 提示词：
    1. 将 ASR 原始对话规范化为完整的书面转写。
    2. 基于规范化内容，提取并生成中文电子病历（EMR）。
    """
    system_msg = (
        "你是一名临床文书与中文电子病历（EMR）规范化专家。\n"
        "请基于提供的口述就诊对话（含ASR转写），严格按顺序完成以下两项任务：\n\n"
        "**任务一：对话规范化**\n"
        "1.  **纠正**：识别并纠正 ASR 误识别、口语化表述、以及不当的断句。\n"
        "2.  **重述**：使用专业的医学术语与标准单位重述（例如：日期/时间、药物剂量、身体部位等）。\n"
        "3.  **输出**：在第一个标题下，输出经过上述处理后的**完整对话全文**。\n\n"
        "4.  **保留角色**：**必须** 严格保留原始对话中的说话人标注（例如 'User A:', 'User B:'），**禁止** 将其替换为任何其他角色。\n"
        "**任务二：EMR结构化提取**\n"
        "1.  **提取**：基于**规范化后**的对话内容，提取信息并逐项填写病历。\n"
        "2.  **约束**：\n"
        "    - 严格保留事实与时序，不进行主观推断或臆造信息。\n"
        "    - 对于对话中确实**未提及**的信息，在相应项目中明确写入“**未提及**”。\n\n"
        "**输出格式**\n"
        "请严格使用以下中文 Markdown 格式，**必须先输出“规范化ASR转写全文”，再输出后续病历清单**：\n\n"
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


def call_deepseek_emr(client: "OpenAI", raw_dialogue: str) -> str:
    """
    调用 DeepSeek 对 ASR 拼接结果进行 EMR 规范化。失败时返回空字符串。
    """
    try:
        messages = _build_emr_prompt(raw_dialogue)
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.error(f"DeepSeek 调用失败：{e}")
        return ""


# ======================= 页面与全局配置 =======================
st.set_page_config(page_title="MedLLM 医疗语音助手", page_icon="🩺", layout="centered")
st.title("🩺 MedLLM 医疗语音助手")


def do_rerun():
    """兼容不同版本 streamlit 的重跑"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# ======================= TTS =======================
def Text_to_audio(Text: str) -> Optional[bytes]:
    """
    返回合成好的 WAV 字节，并用于持久化到对话历史中。
    """
    speed = 1.0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    try:
        model = TTS(language='ZH', device=device)
        speaker_ids = model.hps.data.spk2id
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        model.tts_to_file(Text, speaker_ids['ZH'], out_path, speed=speed)
        with open(out_path, "rb") as f:
            data = f.read()
        try:
            os.unlink(out_path)
        except Exception:
            pass
        return data
    except Exception as e:
        st.warning(f"TTS 失败：{e}")
        return None


# ======================= SenseVoice ASR =======================
@st.cache_resource
def load_sensevoice_model():
    return AutoModel(
        model="iic/SenseVoiceSmall",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        disable_update=True,
        disable_download=False
    )


def asr_from_wav(wav_bytes: bytes) -> str:
    model = load_sensevoice_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(wav_bytes)
        tmp_file_path = tmp_file.name

    try:
        result = model.generate(input=tmp_file_path, batch_size_s=0, language="auto")
        if result and len(result) > 0:
            raw_text = result[0].get("text", "")
            # 清理 <|...|> 等特殊标记
            clean_text = re.sub(r'<\|[^>]+\|>', '', raw_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            print(f"[DEBUG] ASR 原始: {raw_text}")
            print(f"[DEBUG] ASR 清理: {clean_text}")
            return clean_text
        return ""
    except Exception as e:
        print(f"[ASR] 错误: {e}")
        return ""


# ======================= MedLLM =======================
@st.cache_resource
def init_medllm():
    model_path = "/mnt/data/VoiceDoctor/Baichuan2-7B-MedLLM-Merged"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    try:
        gen_cfg = GenerationConfig.from_pretrained(model_path)
        model.generation_config = gen_cfg
    except Exception:
        pass
    try:
        model.generation_config.use_cache = False
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    return model, tokenizer


def medllm_chat(model, tokenizer, messages: List[dict]) -> str:
    if hasattr(model, "chat"):
        return model.chat(tokenizer, messages, stream=False)

    def _join_msgs(msgs):
        segs = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                segs.append(f"[用户]\n{content}")
            else:
                segs.append(f"[助手]\n{content}")
        segs.append("[助手]\n")
        return "\n".join(segs)

    prompt = _join_msgs(messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model = model.to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    last = text.rsplit("[助手]", 1)[-1].strip()
    return last


# ======================= 声纹识别 (SPKREC) =======================

@st.cache_resource
def load_speaker_model():
    """加载 SpeechBrain ECAPA-TDNN 声纹识别模型"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        # 使用 /tmp 目录下的固定路径来缓存模型，避免每次重启都下载
        save_dir = os.path.join("/tmp", "speechbrain_models", "spkrec-ecapa-voxceleb")
        os.makedirs(save_dir, exist_ok=True)
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=save_dir,
            run_opts={"device": device}
        )
        model.eval()
        print(f"声纹识别模型加载成功，运行于 {device}")
        return model
    except Exception as e:
        st.error(f"加载声纹模型失败: {e}")
        return None


def _load_wav_to_tensor(wav_bytes: bytes) -> Optional[torch.Tensor]:
    """从 WAV 字节加载 16k 单声道张量"""
    try:
        with io.BytesIO(wav_bytes) as f:
            wav, sr = torchaudio.load(f)

        # VAD 处理器已经保证了 16k，但以防万一
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # 确保单声道
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        return wav
    except Exception as e:
        print(f"[SPKREC] 从字节加载 WAV 失败: {e}")
        return None


# ======================= recognize_speaker =======================
def recognize_speaker(wav_bytes: bytes, spk_model: EncoderClassifier) -> str:
    """
    识别说话人，如果是新说话人，则分配一个新 ID (User A, B, C...)
    如果匹配到已有说话人，则使用 EMA 更新该说话人的平均 embedding。
    """
    if spk_model is None:
        return "User"  # Fallback

    signal = _load_wav_to_tensor(wav_bytes)
    if signal is None:
        return "User"  # Fallback

    # 检查音频时长，如果太短（<0.5s），embedding 质量极差，跳过识别
    dur_s = signal.shape[1] / 16000.0
    if dur_s < 0.5:
        print(f"[SPKREC] 音频片段过短 ({dur_s:.2f}s)，跳过声纹识别。")
        return "User"  # 返回一个通用 ID

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    signal = signal.to(device)

    try:
        with torch.no_grad():
            # 提取 embedding
            new_embedding = spk_model.encode_batch(signal)
            # (1, 1, 192) -> (192)
            new_embedding = new_embedding.squeeze().cpu()

            # 统一 L2 归一化 (新嵌入)
            new_embedding = torch.nn.functional.normalize(new_embedding, p=2, dim=0)

    except Exception as e:
        print(f"[SPKREC] 提取 embedding 失败: {e}")
        return "User"

    ss = st.session_state

    if not ss.known_speakers:
        # 这是第一个说话人
        new_name = "User A"
        ss.known_speakers.append({
            "name": new_name,
            "embedding": new_embedding,  # 单位向量
            "num_samples": 1
        })
        print(f"[SPKREC] 新增第一个说话人: {new_name}")
        return new_name

    similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    # --- 搜索最佳匹配 ---
    max_sim = -1.0
    best_match_speaker = None

    for speaker in ss.known_speakers:
        known_embedding = speaker["embedding"]
        sim = similarity(new_embedding, known_embedding).item()

        if sim > max_sim:
            max_sim = sim
            best_match_speaker = speaker

    # --- 说话人相似度阈值 ---
    threshold = ss.get("spk_threshold", 0.3)

    if best_match_speaker and max_sim >= threshold:
        # --- 匹配成功：更新平均 Embedding ---
        matched_name = best_match_speaker["name"]
        print(f"[SPKREC] 匹配到: {matched_name} (相似度: {max_sim:.2f})")

        try:
            old_emb = best_match_speaker["embedding"]
            n = best_match_speaker["num_samples"]

            # --- EMA 更新在单位球面上 ---
            ema_alpha = 0.15
            updated_emb = (1 - ema_alpha) * old_emb + ema_alpha * new_embedding
            updated_emb = torch.nn.functional.normalize(updated_emb, p=2, dim=0)

            best_match_speaker["embedding"] = updated_emb
            best_match_speaker["num_samples"] = n + 1
            print(f"[SPKREC] 更新 {matched_name} 的平均 embedding (样本数: {n + 1})")
        except Exception as e:
            print(f"[SPKREC] 更新 embedding 失败: {e}")

        return matched_name

    else:
        # --- 匹配失败：注册新用户 ---
        next_id = len(ss.known_speakers)
        if next_id < 26:
            new_name = f"User {chr(ord('A') + next_id)}"
        else:
            new_name = f"User {chr(ord('A') + (next_id % 26))}{next_id // 26}"

        ss.known_speakers.append({
            "name": new_name,
            "embedding": new_embedding,
            "num_samples": 1
        })
        print(f"[SPKREC] 新增说话人: {new_name} (最高相似度: {max_sim:.2f}, 低于阈值 {threshold})")
        return new_name


# ======================= WebRTC 音频处理器（webrtcvad 分句） =======================
class BrowserAudioProcessor(AudioProcessorBase):
    """
    - 统一重采样为 16k/mono/s16
    - 使用 WebRTC VAD（py-webrtcvad）进行帧级语音活动检测
    - 静音累计 >= 0.5s 视为一句结束，推入 ready_segments（WAV bytes）
    - prepad 120ms 进入一句时补上前导，tail 保留 500ms
    - 上层 UI 会逐条展示每个片段（文本 + 音频）
    """

    def __init__(self):
        # 输出采样设为 16k，以满足 webrtcvad 支持的采样率
        self.sr_out = 16000
        self.resampler = AudioResampler(format="s16", layout="mono", rate=self.sr_out)

        # WebRTC VAD：0(宽松)~3(激进)，这里取 2，实测口语对话比较稳健
        self.vad = webrtcvad.Vad(2)

        self.is_collecting = False
        self.input_rate = None

        # webrtcvad 仅支持 10/20/30ms 帧；保持 20ms
        self.frame_ms = 20
        self.frame_len = int(self.sr_out * self.frame_ms / 1000)

        # 句子边界 & 过滤
        self.silence_limit_s = 0.5
        self.min_utt_s = 0.35

        # 进入一句时的前置缓冲 & 句末保留
        self.prepad_ms = 120
        self.prepad_len = int(self.sr_out * self.prepad_ms / 1000)
        self.tail_keep_ms = 500
        self.tail_keep_len = int(self.sr_out * self.tail_keep_ms / 1000)

        # 缓存与状态
        self._carry = np.zeros(0, dtype=np.int16)
        self._pre_buffer = np.zeros(0, dtype=np.int16)
        self._utter_active = False
        self._utter_pcm = np.zeros(0, dtype=np.int16)
        self._silence_accum_s = 0.0

        self._ready_segments: List[bytes] = []

    def reset(self):
        self._carry = np.zeros(0, dtype=np.int16)
        self._pre_buffer = np.zeros(0, dtype=np.int16)
        self._utter_active = False
        self._utter_pcm = np.zeros(0, dtype=np.int16)
        self._silence_accum_s = 0.0

    def start_collect(self):
        self.reset()
        self.is_collecting = True
        print("[DEBUG] 开始采集（等待说话触发首句）...")

    def stop_collect(self):
        self.is_collecting = False
        print("[DEBUG] 停止采集。输入采样率:", self.input_rate)

    def _export_wav_bytes(self, pcm: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr_out)
            wf.writeframes(pcm.astype(np.int16, copy=False).tobytes())
        return buf.getvalue()

    def _vad_is_speech(self, frame_i16: np.ndarray) -> bool:
        """
        使用 webrtcvad 判断单帧是否为语音。
        传入参数必须是 16k/单声道/16-bit PCM，长度=10/20/30ms。
        """
        try:
            # 必须是 bytes（little-endian 16-bit PCM）
            return self.vad.is_speech(frame_i16.tobytes(), sample_rate=self.sr_out)
        except Exception as e:
            # 捕获异常避免中断音频管线
            print(f"[VAD] is_speech 异常: {e}")
            return False

    def _finalize_utter_if_needed(self, force: bool = False):
        if self._utter_active and (force or self._silence_accum_s >= self.silence_limit_s):
            pcm = self._utter_pcm
            if self.tail_keep_len > 0 and self._carry.size > 0:
                tail = self._carry[:self.tail_keep_len]
                pcm = np.concatenate([pcm, tail], axis=0)

            dur = pcm.size / float(self.sr_out)
            if dur >= self.min_utt_s:
                wav_bytes = self._export_wav_bytes(pcm)
                self._ready_segments.append(wav_bytes)
                print(f"[VAD] 句子完成: {dur:.2f}s, 队列数={len(self._ready_segments)}")
            else:
                print(f"[VAD] 丢弃过短片段: {dur:.2f}s")

            self._utter_active = False
            self._utter_pcm = np.zeros(0, dtype=np.int16)
            self._silence_accum_s = 0.0
            self._pre_buffer = np.zeros(0, dtype=np.int16)

    def force_flush_current_utterance(self):
        self._finalize_utter_if_needed(force=True)

    def pop_ready_segment_wav(self) -> Optional[bytes]:
        if self._ready_segments:
            return self._ready_segments.pop(0)
        return None

    def _frame_to_int16_mono_16k(self, frame: av.AudioFrame) -> List[np.ndarray]:
        out = []
        if self.input_rate is None:
            try:
                fmt = frame.format.name if frame.format else "unknown"
                layout = frame.layout.name if frame.layout else "unknown"
            except Exception:
                fmt, layout = "unknown", "unknown"
            self.input_rate = getattr(frame, "sample_rate", None)
            print(f"[DEBUG] 检测到输入采样率: {self.input_rate}, 格式: {fmt}, 布局: {layout}")

        try:
            out_frames = self.resampler.resample(frame) or []
        except Exception as e:
            print(f"[ERROR] 重采样失败: {e}")
            return out

        for f in out_frames:
            arr = f.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            arr = np.asarray(arr, dtype=np.int16).reshape(-1)
            out.append(arr)
        return out

    def _process_pcm_for_vad(self, pcm: np.ndarray):
        cat = pcm if self._carry.size == 0 else np.concatenate([self._carry, pcm], axis=0)
        n = (cat.size // self.frame_len) * self.frame_len
        frames = cat[:n].reshape(-1, self.frame_len)
        self._carry = cat[n:]

        for fr in frames:
            is_speech = self._vad_is_speech(fr)

            if not self._utter_active:
                # 维护进入一句的前置缓冲（最多 prepad_len）
                if self._pre_buffer.size == 0:
                    self._pre_buffer = fr.copy()
                else:
                    self._pre_buffer = np.concatenate([self._pre_buffer, fr], axis=0)[-self.prepad_len:]

                if is_speech and self.is_collecting:
                    self._utter_active = True
                    # 带上 prepad 提前量，避免首音爆破被切掉
                    if self._pre_buffer.size > 0:
                        self._utter_pcm = self._pre_buffer.copy()
                    else:
                        self._utter_pcm = np.zeros(0, dtype=np.int16)
                    self._utter_pcm = np.concatenate([self._utter_pcm, fr], axis=0)
                    self._silence_accum_s = 0.0
            else:
                self._utter_pcm = np.concatenate([self._utter_pcm, fr], axis=0)
                if is_speech:
                    self._silence_accum_s = 0.0
                else:
                    self._silence_accum_s += self.frame_ms / 1000.0
                    if self._silence_accum_s >= self.silence_limit_s:
                        self._finalize_utter_if_needed(force=False)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            chunks = self._frame_to_int16_mono_16k(frame)
            if self.is_collecting and chunks:
                for ch in chunks:
                    self._process_pcm_for_vad(ch)
        except Exception as e:
            print(f"[ERROR] 处理音频帧失败: {e}")
        return frame


# ======================= 会话状态 =======================
def ensure_state():
    ss = st.session_state
    ss.setdefault("messages", [])  # 每条可包含 {"role","content","audio"(bytes|None)}
    ss.setdefault("rec_active", False)
    # asr_segments: List[Dict]，每条包含 {"text": str, "wav": bytes, "speaker": str}
    ss.setdefault("asr_segments", [])
    ss.setdefault("need_loop", False)  # 控制轮询

    # --- 声纹识别状态 ---
    ss.setdefault("known_speakers", [])

    # --- 阈值 ---
    ss.setdefault("spk_threshold", 0.3)

    # --- 保存最近一次 DeepSeek 规范化病历 ---
    ss.setdefault("last_emr_text", "")


ensure_state()


def _append_message(role: str, content: str, audio: Optional[bytes] = None):
    """
    统一入口：仅写入 state，不做即时 UI 渲染；并做简单去重，避免重复 append。
    """
    msgs = st.session_state["messages"]
    if msgs:
        last = msgs[-1]
        if last.get("role") == role and (last.get("content") or "") == (content or ""):
            return
    st.session_state["messages"].append({"role": role, "content": content, "audio": audio})


# ======================= 启动即初始化 ASR、LLM、SPKREC、DeepSeek =======================
with st.spinner("加载 ASR 模型中…"):
    _ = load_sensevoice_model()
with st.spinner("加载 MedLLM 模型中…"):
    model, tokenizer = init_medllm()
with st.spinner("加载声纹识别模型中…"):
    speaker_model = load_speaker_model()
with st.spinner("初始化 DeepSeek API 工具…"):
    ds_client = init_deepseek_client()

# ======================= 滚动容器：聊天历史 =======================
st.subheader("🗨️ 对话历史")
chat_box = st.container(height=520, border=True)
with chat_box:
    if not st.session_state["messages"]:
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown("您好，我是 MedLLM 医疗助手大模型，很高兴为您服务。")
    for m in st.session_state["messages"]:
        avatar = "🧑‍💻" if m["role"] == "user" else "🩺"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m.get("content", ""))
            if m.get("audio"):
                st.audio(m["audio"], format="audio/wav")

# ======================= 录音 UI =======================
st.subheader("🎤 浏览器录音（Start → webrtcvad 自动分句 → 实时转写+音频预览 → DeepSeek一键转译 → 输入AI医生）")
st.caption(
    "操作：先点击上方内置 **Start** → 点击 **开始录音** → 讲话；使用 **WebRTC VAD** 检测，静音≥0.5秒自动切句；"
    "每句会显示 **[说话人ID]** + 转写 + 该段音频；"
    "点击 **DeepSeek一键转译** 结束录音并生成规范化病历；如需继续问诊，点击 **输入AI医生**。"
)

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        # --- 开启浏览器端 3A 处理 ---
        "audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True},
        "video": False
    },
    audio_processor_factory=BrowserAudioProcessor,
    async_processing=False,
)

# 三列按钮：开始录音、DeepSeek一键转译、输入AI医生
col_rec1, col_rec2, col_rec3 = st.columns(3)
with col_rec1:
    start_btn = st.button("开始录音 ▶️", type="primary", use_container_width=True)
with col_rec2:
    ds_btn = st.button("DeepSeek一键转译 🧠", use_container_width=True)
with col_rec3:
    send_llm_btn = st.button("输入AI医生 🤖", use_container_width=True)

# -------------- 事件：DeepSeek一键转译（结束录音并规范化病历） --------------
if ds_btn:
    if ctx and ctx.audio_processor and st.session_state["rec_active"]:
        # 停止采集并强制冲刷尾句
        ctx.audio_processor.stop_collect()
        ctx.audio_processor.force_flush_current_utterance()
        st.session_state["need_loop"] = False

        # Drain 所有待处理片段
        popped_items: List[Dict] = []
        while True:
            seg = ctx.audio_processor.pop_ready_segment_wav()
            if seg is None:
                break
            # 检查时长
            try:
                with io.BytesIO(seg) as f, wave.open(f, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    dur_s = frames / float(rate)
                if dur_s < 1.0:
                    print(f"[FILTER] 丢弃过短片段: {dur_s:.2f}s (阈值 1.0s)")
                    continue
            except Exception as e:
                print(f"[FILTER] 检查音频时长失败: {e}")
                continue

            # 声纹 + ASR
            speaker_id = recognize_speaker(seg, speaker_model)
            txt = asr_from_wav(seg)
            if txt:
                formatted_text = f"{speaker_id}: {txt}"  # 去掉 markdown 粗体，更利于后续处理
                item = {"text": formatted_text, "wav": seg, "speaker": speaker_id}
                popped_items.append(item)
            else:
                print(f"[FILTER] 丢弃空转写结果 (来自 {speaker_id})")

        # 把最后冲刷出来的片段更新到 UI 状态中（用于“实时转写”展示的一致性）
        st.session_state["asr_segments"].extend(popped_items)

        # 构造原始对话串（逐行）
        raw_lines = [it.get("text", "") for it in st.session_state.get("asr_segments", []) if it.get("text")]
        raw_dialogue = "\n".join(raw_lines).strip()

        # 重置采集器与轮询
        ctx.audio_processor.reset()
        st.session_state["rec_active"] = False
        if hasattr(ctx, "stop"):
            try:
                ctx.stop()
            except:
                pass
        elif hasattr(ctx, "request_stop"):
            try:
                ctx.request_stop()
            except:
                pass

        # 清空临时转写与声纹库（这一轮已完成）
        st.session_state["asr_segments"] = []
        st.session_state["known_speakers"] = []

        if not raw_dialogue:
            st.info("未检测到有效语音内容，本次未发送至 DeepSeek。")
        else:
            if ds_client is None:
                st.error("DeepSeek API 工具不可用，请检查 openai 安装与 DEEPSEEK_API_KEY。")
            else:
                with st.spinner("DeepSeek 正在规范化病历…"):
                    emr_text = call_deepseek_emr(ds_client, raw_dialogue)

                if emr_text:
                    # 保存到 session 并输出到对话历史（assistant 身份展示“规范化病历”）
                    st.session_state["last_emr_text"] = emr_text
                    _append_message(
                        "assistant",
                        "### 📋 DeepSeek 规范化病历\n\n" + emr_text
                    )
                    do_rerun()
                else:
                    st.error("未得到有效的规范化病历文本。")
    else:
        st.warning("当前没有正在进行的录音。请先点击『开始录音』。")

# -------------- 事件：输入AI医生（把最近一次 DeepSeek 规范化病历送入 MedLLM 并 TTS） --------------
if send_llm_btn:
    emr_text = st.session_state.get("last_emr_text", "").strip()
    if not emr_text:
        st.warning("没有可用的规范化病历文本，请先执行『DeepSeek一键转译』。")
    else:
        # 1. 构造一个临时的消息列表，包含当前历史和即将发送的 EMR 文本
        messages_for_llm = st.session_state["messages"].copy()
        messages_for_llm.append({"role": "user", "content": emr_text})

        try:
            # 2. 使用这个临时列表来调用 LLM
            reply = medllm_chat(model, tokenizer, messages_for_llm)
        except Exception as e:
            reply = f"对话失败：{e}"

        # 3. LLM 的回复被正常追加到 *永久* 对话历史中
        audio_bytes = Text_to_audio(Text=reply)
        _append_message("assistant", reply, audio=audio_bytes)
        do_rerun()

# -------------- 事件：开始录音 --------------
if start_btn:
    if ctx and ctx.state.playing:
        if ctx.audio_processor:
            ctx.audio_processor.start_collect()
            st.session_state["rec_active"] = True
            # 清空上一轮状态
            st.session_state["asr_segments"] = []
            st.session_state["known_speakers"] = []
            st.session_state["need_loop"] = True
            st.info("正在录音…（WebRTC VAD：静音≥0.5秒自动切句；每句附带音频预览）")
        else:
            st.error("音频处理器未就绪，请刷新页面重试。")
    else:
        st.warning("请先点击上方内置的 **Start** 按钮以建立麦风连接（浏览器会弹出权限请求）。")


# -------------- 实时 drain：句段转写（带声纹） --------------
def _drain_ready_segments_into_state() -> int:
    """
    修改后的版本：
    - 每次只处理队列中的 *一个* 片段，并立即返回。
    - 返回值：1 (处理成功1个), 0 (队列为空或处理失败)
    """
    if not (ctx and ctx.audio_processor):
        return 0

    # 1. 每次只尝试 Pop 一个片段
    seg = ctx.audio_processor.pop_ready_segment_wav()
    if seg is None:
        # 队列为空，返回 0
        return 0

    # 2. 检查时长
    try:
        with io.BytesIO(seg) as f, wave.open(f, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            dur_s = frames / float(rate)
        if dur_s < 1.0:
            print(f"[FILTER] 丢弃过短片段: {dur_s:.2f}s (阈值 1.0s)")
            return 0  # 算作处理了，但不算有效结果
    except Exception as e:
        print(f"[FILTER] 检查音频时长失败: {e}")
        return 0

    # 3. 声纹 + ASR
    speaker_id = recognize_speaker(seg, speaker_model)
    text = asr_from_wav(seg)

    if text:
        formatted_text = f"{speaker_id}: {text}"
        item = {"text": formatted_text, "wav": seg, "speaker": speaker_id}
        st.session_state["asr_segments"].append(item)
        print(f"[UI] 处理完成 1 个句段: {formatted_text}")
        return 1  # 成功处理 1 个
    else:
        print(f"[FILTER] 丢弃空转写结果 (来自 {speaker_id})")
        return 0  # 算作处理了，但不算有效结果


# ======================= 实时转写区域 =======================
if len(st.session_state["asr_segments"]) > 0:
    st.subheader("📝 实时转写")
    asr_box = st.container(height=360, border=True)
    with asr_box:
        for item in st.session_state["asr_segments"]:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(item.get("text", ""))
                wav_bytes = item.get("wav", None)
                if wav_bytes:
                    st.audio(wav_bytes, format="audio/wav")

# -------------- 轮询刷新（仅在录音中） --------------
if st.session_state["rec_active"] and st.session_state.get("need_loop", False):

    # 1. 尝试从音频处理器中获取并处理（ASR+SPKREC） *一个* 已就绪的片段
    processed_count = _drain_ready_segments_into_state()

    # 2. 检查是否有新结果
    if processed_count > 0:
        # 2a. 如果处理得到了1个新片段 (ASR有结果)，立即重绘UI，不睡眠
        #     (do_rerun() 之后会马上再次进入此循环，继续处理队列中可能的下一个片段)
        do_rerun()
    else:
        # 2b. 如果没有新片段 (VAD队列为空)，短暂睡眠 (例如 100ms)
        #     以避免CPU空转，然后再重绘以继续轮询
        time.sleep(0.1)
        do_rerun()

# ======================= 备用：文本输入发送 =======================
with st.expander("✍️ 也可直接输入文本"):
    col_input, col_send = st.columns([8, 2])
    with col_input:
        text_input = st.text_input("输入文本（Enter 发送）", key="text_input")
    with col_send:
        send_btn = st.button("发送", use_container_width=True)

    if send_btn and text_input.strip():
        u = text_input.strip()
        _append_message("user", u)
        try:
            reply = medllm_chat(model, tokenizer, st.session_state["messages"])
        except Exception as e:
            reply = f"对话失败：{e}"
        audio_bytes = Text_to_audio(Text=reply)
        _append_message("assistant", reply, audio=audio_bytes)
        do_rerun()

# ======================= 底部工具栏（单列） =======================
st.divider()


def _clear_chat():
    st.session_state["messages"] = []
    st.session_state["asr_segments"] = []
    st.session_state["need_loop"] = False
    st.session_state["known_speakers"] = []
    st.session_state["last_emr_text"] = ""
    if 'ctx' in globals() and ctx and ctx.audio_processor:
        ctx.audio_processor.reset()


st.button("🧹 清空对话", on_click=_clear_chat, use_container_width=True)
