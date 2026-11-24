"""Provider-neutral JSON transport with bounded timeouts and offline fallback."""
import base64
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        result = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in model response")
        result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


class LLMClient:
    def _post(self, url, body, headers=None, timeout=25):
        request = Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "AutoLens/1.0",
                **(headers or {}),
            },
        )
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def complete(self, system, prompt, image=None, mime=None):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        encoded = base64.b64encode(image).decode() if image else None
        providers = []
        if os.getenv("OPENROUTER_API_KEY"):
            providers.append("openrouter")
        if os.getenv("GEMINI_API_KEY"):
            providers.append("gemini")
        if not image:
            providers.append("ollama")

        for provider in providers:
            try:
                if provider == "openrouter":
                    content = (
                        prompt
                        if not image
                        else [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{encoded}"},
                            },
                        ]
                    )
                    model = os.getenv(
                        "OPENROUTER_VISION_MODEL" if image else "OPENROUTER_MODEL"
                    ) or "openrouter/free"
                    response = self._post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [
                                messages[0],
                                {"role": "user", "content": content},
                            ],
                            "temperature": 0.2,
                        },
                        {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                    )
                    text = response["choices"][0]["message"]["content"]
                elif provider == "gemini":
                    parts = [{"text": prompt}]
                    if image:
                        parts.append({"inlineData": {"mimeType": mime, "data": encoded}})
                    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                    response = self._post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        {
                            "systemInstruction": {"parts": [{"text": system}]},
                            "contents": [{"role": "user", "parts": parts}],
                            "generationConfig": {
                                "temperature": 0.2,
                                "responseMimeType": "application/json",
                            },
                        },
                        {"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                    )
                    text = "".join(
                        p.get("text", "")
                        for p in response["candidates"][0]["content"]["parts"]
                    )
                else:
                    response = self._post(
                        "http://localhost:11434/api/chat",
                        {
                            "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                            "messages": messages,
                            "stream": False,
                            "format": "json",
                            "options": {"temperature": 0},
                        },
                        timeout=3,
                    )
                    text = response["message"]["content"]
                return parse_json(text)
            except HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="ignore")[:300]
                print(f"[LLMClient] {provider} HTTP {exc.code}: {err_body}")
                continue
            except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                print(f"[LLMClient] {provider} failed: {exc}")
                continue
        return None