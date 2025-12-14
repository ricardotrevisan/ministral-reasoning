import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()


class SerperClient:
    """
    Cliente simples para a API do Serper.
    Espera as variáveis de ambiente:
      - SERPER_API_KEY
      - SERPER_BASE_URL (ex.: https://google.serper.dev)
      - SEARCH_TIMEOUT (opcional, em segundos)
    """

    def __init__(self) -> None:
        api_key = os.getenv("SERPER_API_KEY")
        base_url = os.getenv("SERPER_BASE_URL", "").rstrip("/")

        if not api_key:
            raise ValueError("SERPER_API_KEY não configurado.")
        if not base_url:
            raise ValueError("SERPER_BASE_URL não configurado.")

        self.api_key = api_key
        self.base_url = base_url
        self.timeout = int(os.getenv("SEARCH_TIMEOUT", "20"))

    def search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Faz uma busca simples usando o Serper e retorna
        uma lista de dicionários com pelo menos:
          - title
          - url
          - snippet
        """
        payload: Dict[str, Any] = {
            "q": query,
            "num": max_results,
        }

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/search"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        organic = data.get("organic") or []
        results: List[Dict[str, Any]] = []

        for item in organic[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link") or item.get("url", ""),
                    "snippet": item.get("snippet") or item.get("description", ""),
                }
            )

        return results

