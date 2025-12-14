import json
from typing import Any, Dict, List

import requests

from utils.search_tool import WebSearchTool

URL = "http://127.0.0.1:8080/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

#MODEL_NAME = "Ministral-3-8B-Reasoning-2512-Q5_K_M.gguf"
MODEL_NAME = "local-container-defined"

SYSTEM_PROMPT = (
    "Você é um assistente que pode chamar funções (tools) quando necessário. "
    "Use a ferramenta de busca na web quando precisar de informações atualizadas "
    "ou fatos externos, em vez de inventar respostas."
)


# Instância única da nossa tool de busca
web_search_tool = WebSearchTool()


FUNCTIONS: Dict[str, Any] = {
    web_search_tool.name: web_search_tool.run,
}


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": web_search_tool.name,
            "description": web_search_tool.description,
            "parameters": web_search_tool.parameters,
        },
    }
]


def call_model(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = requests.post(URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]


def stream_final_response(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Segunda chamada: mesma API, mas com stream=True para receber
    a resposta final do assistente token a token.
    """
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "stream": True,
    }

    # Nesta fase não queremos novas tool_calls, só a resposta final.
    with requests.post(URL, headers=HEADERS, json=payload, stream=True) as r:
        full_content = ""
        for raw_line in r.iter_lines():
            if not raw_line:
                continue

            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue

            content = line[len("data:") :].strip()
            if content == "[DONE]":
                break

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue

            delta = data["choices"][0]["delta"]
            token = delta.get("content") or ""
            if token:
                print(token, end="", flush=True)
                full_content += token

    print()  # quebra de linha após o streaming
    return {"role": "assistant", "content": full_content}


def handle_tool_calls(
    message: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> None:
    """Executa todas as tool_calls solicitadas pelo modelo."""
    tool_calls = message.get("tool_calls") or []

    for tool_call in tool_calls:
        func_name = tool_call["function"]["name"]
        func = FUNCTIONS.get(func_name)
        arguments_str = tool_call["function"].get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            arguments = {}

        if func is None:
            result = f"Ferramenta desconhecida: {func_name}"
        else:
            # Aqui as ferramentas recebem argumentos já parseados (dict).
            result = func(arguments)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "name": func_name,
                "content": str(result),
            }
        )


def chat_with_tools() -> None:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print("🟢 Chat com tools iniciado. Digite sua pergunta. ('sair' para encerrar)\n")

    while True:
        user_input = input("\nVocê: ").strip()

        if user_input.lower() in ("sair", "exit", "quit", "bye"):
            print("🔴 Chat encerrado.")
            break

        messages.append({"role": "user", "content": user_input})

        # 1ª chamada: modelo decide se usa tool
        choice = call_model(messages, tools=TOOLS)
        message = choice["message"]

        # Se o modelo pediu ferramentas, executa e faz 2ª chamada
        if message.get("tool_calls"):
            messages.append(message)
            handle_tool_calls(message, messages)

            # 2ª chamada: agora com as respostas das tools (streaming)
            print("Assistente: ", end="", flush=True)
            message = stream_final_response(messages)
        else:
            # Sem tools: resposta direta (não-streaming)
            choice = call_model(messages, tools=TOOLS)
            message = choice["message"]
            assistant_content = message.get("content", "").strip()
            if assistant_content:
                print(f"Assistente: {assistant_content}")
                messages.append({"role": "assistant", "content": assistant_content})
            continue

        # Exibe e salva a mensagem já construída no fluxo de streaming
        assistant_content = message.get("content", "").strip()
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})


if __name__ == "__main__":
    chat_with_tools()
