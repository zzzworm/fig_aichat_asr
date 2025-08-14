import warnings
import torch
import os

try:
    import openai_whisper as whisper
except ImportError:
    try:
        import whisper
    except ImportError:
        print("Error: Neither openai-whisper nor whisper package found.")
        print("Please install with: pip install openai-whisper")
        raise ImportError("Whisper package not found")

class WhisperASR:
    def __init__(self, model_name="small"):
        """
        初始化Whisper ASR模型
        
        Args:
            model_name: 模型大小 ("tiny", "base", "small", "medium", "large")
        """
        try:
            # 检查设备可用性
            self.device = self._get_device()
            print(f"Using device: {self.device}")
            
            # 加载模型
            print(f"Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name, device=self.device)
            print("Whisper model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
    
    def _get_device(self):
        """
        获取可用的计算设备
        
        Returns:
            str: 设备名称 ("cuda" 或 "cpu")
        """
        if torch.cuda.is_available():
            return "cuda"
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     # 对于 Apple Silicon Mac
        #     return "mps"
        else:
            return "cpu"
    
    def transcribe(self, audio_path):
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            dict: 包含转录结果的字典
        """
        try:
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # 设置转录选项，避免警告
            options = {
                "task": "transcribe",  # 明确指定任务为转录
                "language": None,      # 自动检测语言
                "verbose": False,      # 减少输出信息
                "temperature": 0.0,    # 设置温度参数提高稳定性
                "compression_ratio_threshold": 2.4,  # 压缩比阈值
                "logprob_threshold": -1.0,           # 对数概率阈值
                "no_speech_threshold": 0.6,          # 无语音阈值
            }
            
            # 根据设备类型调整参数
            if self.device == "cuda":
                options["fp16"] = True  # GPU 可以使用 FP16
            else:
                options["fp16"] = False  # CPU 使用 FP32
            
            # 忽略特定警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
                warnings.filterwarnings("ignore", message=".*attention_mask.*")
                warnings.filterwarnings("ignore", message=".*pad token.*")
                warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
                
                # 进行转录
                print(f"Starting transcription for: {audio_path}")
                result = self.model.transcribe(audio_path, **options)
                print(f"Transcription completed")
            
            # 提取转录文本并清理
            transcription = result.get("text", "").strip()
            
            # 返回标准格式
            return {
                "transcription": transcription,
                "language": result.get("language", "unknown"),
                "confidence": self._calculate_confidence(result),
                "segments": result.get("segments", []),
                "processing_info": {
                    "model": model_name if hasattr(self, 'model_name') else "whisper",
                    "device": self.device,
                    "audio_duration": self._get_audio_duration(result),
                    "detected_language": result.get("language", "unknown")
                }
            }
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {
                "transcription": "",
                "error": str(e),
                "language": "unknown",
                "confidence": 0.0,
                "processing_info": {
                    "device": self.device,
                    "error": str(e)
                }
            }
    
    def _calculate_confidence(self, result):
        """
        计算平均置信度
        
        Args:
            result: Whisper转录结果
            
        Returns:
            float: 平均置信度
        """
        try:
            segments = result.get("segments", [])
            if not segments:
                return 0.0
            
            # 计算所有段的平均置信度
            total_confidence = 0.0
            total_duration = 0.0
            
            for segment in segments:
                duration = segment.get("end", 0) - segment.get("start", 0)
                confidence = segment.get("avg_logprob", 0.0)
                # 将对数概率转换为置信度 (0-1)
                confidence = max(0.0, min(1.0, (confidence + 1.0) / 2.0))
                
                total_confidence += confidence * duration
                total_duration += duration
            
            return total_confidence / total_duration if total_duration > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _get_audio_duration(self, result):
        """
        获取音频时长
        
        Args:
            result: Whisper转录结果
            
        Returns:
            float: 音频时长（秒）
        """
        try:
            segments = result.get("segments", [])
            if segments:
                return segments[-1].get("end", 0.0)
            return 0.0
        except Exception:
            return 0.0