"""OpenRouter API client."""

import base64
import json
import struct
import time
import wave
from pathlib import Path

import httpx

BASE_URL = "https://openrouter.ai/api/v1"
_RATE_LIMIT_RETRIES = 4
_RATE_LIMIT_BASE_DELAY = 1.0  # seconds


class OpenRouterClient:
    def __init__(self, api_key: str, timeout: float = 120.0):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/somebox/storytime",
        }

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a chat completion request. Returns the full response dict."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(_RATE_LIMIT_RETRIES + 1):
                resp = client.post(
                    f"{BASE_URL}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                if resp.status_code == 429 and attempt < _RATE_LIMIT_RETRIES:
                    delay = float(resp.headers.get("retry-after", _RATE_LIMIT_BASE_DELAY * (2 ** attempt)))
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()

    def chat_text(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, dict]:
        """Send a chat completion and return (text, usage_dict).

        usage_dict has keys: input_tokens, output_tokens, cost (all ints/floats,
        default 0) and served_model (the concrete model that served the request).
        """
        data = self.chat(model, messages, temperature, max_tokens)
        content = data["choices"][0]["message"]["content"] or ""
        raw_usage = data.get("usage") or {}
        usage = {
            "input_tokens": raw_usage.get("prompt_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
            "cost": float(data.get("usage", {}).get("cost", 0) or 0),
            # With `openrouter/auto` the router picks a concrete model and reports
            # it back in the top-level `model` field; for a pinned request it
            # echoes the requested id. Fall back to the requested model if absent.
            "served_model": str(data.get("model") or model),
        }
        return content, usage

    def describe_image(
        self,
        model: str,
        image_url: str = "",
        image_bytes: bytes = b"",
        mime_type: str = "image/jpeg",
    ) -> str:
        """Describe an image using a vision-capable model.

        Accepts either an external URL or raw bytes (converted to data URI).
        Returns a short text description.
        """
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode()
            url = f"data:{mime_type};base64,{b64}"
        else:
            url = image_url

        content = [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": (
                "Describe this image in 1-2 sentences for an art director. "
                "Focus on: subject, composition, lighting, color/tone, era, "
                "and photographic style. Be specific and concise."
            )},
        ]
        data = self.chat(
            model, [{"role": "user", "content": content}],
            temperature=0.3, max_tokens=200,
        )
        return (data["choices"][0]["message"]["content"] or "").strip()

    def chat_audio(
        self,
        model: str,
        messages: list[dict],
        voice: str = "sage",
        audio_format: str = "wav",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[bytes, str]:
        """Stream a chat completion with audio output.

        Streaming requires pcm16 format. The raw PCM is converted to the
        requested format (wav by default) before returning.

        Returns (audio_bytes, transcript).
        """
        # Streaming only supports pcm16 — we convert after
        payload = {
            "model": model,
            "messages": messages,
            "modalities": ["text", "audio"],
            "audio": {"voice": voice, "format": "pcm16"},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        audio_chunks: list[str] = []
        transcript_chunks: list[str] = []

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{BASE_URL}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    # Read the error body before raising
                    error_body = ""
                    for chunk in resp.iter_text():
                        error_body += chunk
                    raise httpx.HTTPStatusError(
                        f"{resp.status_code}: {error_body}",
                        request=resp.request,
                        response=resp,
                    )
                for line in resp.iter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # Audio data comes as base64 chunks
                    audio_data = delta.get("audio", {})
                    if isinstance(audio_data, dict):
                        if "data" in audio_data:
                            audio_chunks.append(audio_data["data"])
                        if "transcript" in audio_data:
                            transcript_chunks.append(audio_data["transcript"])

        pcm_bytes = base64.b64decode("".join(audio_chunks)) if audio_chunks else b""
        transcript = "".join(transcript_chunks)

        # Convert raw PCM16 to requested format
        if pcm_bytes and audio_format == "wav":
            audio_bytes = _pcm16_to_wav(pcm_bytes)
        else:
            # For other formats or empty, return raw PCM
            audio_bytes = pcm_bytes

        return audio_bytes, transcript

    def chat_audio_to_file(
        self,
        model: str,
        messages: list[dict],
        output_path: Path,
        voice: str = "sage",
        audio_format: str = "wav",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Stream audio to a file. Returns the transcript."""
        audio_bytes, transcript = self.chat_audio(
            model=model,
            messages=messages,
            voice=voice,
            audio_format=audio_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)
        return transcript


    # Models known to accept image input AND produce image output.
    # Checked against OpenRouter /api/v1/models (input_modalities includes "image").
    _MULTIMODAL_PREFIXES = ("google/gemini", "openai/gpt")

    def _supports_image_input(self, model: str) -> bool:
        """Check if a model supports reference images as input.

        Gemini and GPT image models accept multimodal input (text + images)
        and output (text + images). Pure generators like FLUX do not.
        """
        return any(model.startswith(p) for p in self._MULTIMODAL_PREFIXES)

    def chat_image(
        self,
        model: str,
        prompt: str,
        reference_urls: list[str] | None = None,
        aspect_ratio: str = "16:9",
    ) -> bytes | None:
        """Generate an image via chat completion.

        If reference_urls are provided AND the model supports image input,
        they are included so the model can use them as visual reference.
        Models that only generate images (e.g., FLUX) use text-only prompts.

        Returns image bytes (PNG) or None if no image in response.
        """
        supports_refs = self._supports_image_input(model)

        # Build multimodal content only if model supports it
        if reference_urls and supports_refs:
            content = []
            for url in reference_urls:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt

        # Gemini models support both text+image output;
        # pure image models (FLUX, Sourceful) only output images
        if supports_refs:
            modalities = ["image", "text"]
        else:
            modalities = ["image"]

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "modalities": modalities,
            "image_config": {"aspect_ratio": aspect_ratio},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{BASE_URL}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return _extract_image(data)

    def chat_image_to_file(
        self,
        model: str,
        prompt: str,
        output_path: Path,
        reference_urls: list[str] | None = None,
        aspect_ratio: str = "16:9",
    ) -> bool:
        """Generate an image and save to file. Returns True if successful."""
        image_bytes = self.chat_image(
            model, prompt, reference_urls=reference_urls,
            aspect_ratio=aspect_ratio,
        )
        if image_bytes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            suffix = _image_bytes_format_suffix(image_bytes)
            dest = output_path.with_suffix(suffix)
            stem = output_path.with_suffix("")
            if dest != output_path and output_path.exists():
                output_path.unlink()
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                other = stem.with_suffix(ext)
                if other != dest and other.exists():
                    other.unlink()
            dest.write_bytes(image_bytes)
            return True
        return False


def _image_bytes_format_suffix(data: bytes) -> str:
    """Pick ``.png``, ``.jpg``, or ``.webp`` from magic bytes (API may return JPEG)."""

    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _extract_image(data: dict) -> bytes | None:
    """Extract image bytes from an OpenRouter image generation response."""
    message = data["choices"][0]["message"]

    # Check for images array
    images = message.get("images", [])
    if images:
        for img in images:
            url = img.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                b64_data = url.split(",", 1)[1]
                return base64.b64decode(b64_data)

    # Check content array for inline image parts
    content = message.get("content", "")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    b64_data = url.split(",", 1)[1]
                    return base64.b64decode(b64_data)
            if part.get("type") == "inline_data":
                b64_data = part.get("data", "")
                if b64_data:
                    return base64.b64decode(b64_data)

    return None


AUDIO_SAMPLE_RATE = 24000  # OpenAI audio models output 24kHz
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


def _pcm16_to_wav(pcm_bytes: bytes) -> bytes:
    """Convert raw PCM16 mono 24kHz audio to WAV format."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(AUDIO_SAMPLE_WIDTH)
        wf.setframerate(AUDIO_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _parse_sse_line(line: str) -> dict | None:
    """Parse a single SSE line into a dict, or None if not a data line."""
    line = line.strip()
    if not line or not line.startswith("data: "):
        return None
    data = line[6:]
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None
