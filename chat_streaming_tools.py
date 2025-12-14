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


def call_model(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
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


def handle_tool_calls(message: Dict[str, Any]) -> str:
    """
    Executa todas as tool_calls solicitadas pelo modelo e
    retorna um texto resumindo os resultados para o LLM.
    Não adiciona mensagens com role=tool ao histórico, pois
    o servidor não lida bem com esse role.
    """
    tool_calls = message.get("tool_calls") or []

    results_for_llm: List[str] = []

    for tool_call in tool_calls:
        func_name = tool_call["function"]["name"]
        func = FUNCTIONS.get(func_name)
        arguments_str = tool_call["function"].get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            arguments = {}

        print(f"[tool_call] {func_name} args={arguments}")

        if func is None:
            result = f"Ferramenta desconhecida: {func_name}"
        else:
            result = func(arguments)

        print(f"[tool_result] {func_name} -> {result}")
        results_for_llm.append(f"{func_name}: {result}")

    return "\n".join(results_for_llm)


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

        # adiciona a pergunta do usuário ao histórico
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # 1ª chamada: modelo decide se usa tool
        choice = call_model(messages, tools=TOOLS)
        message = choice["message"]

        # Se o modelo pediu ferramentas, executa e faz 2ª chamada
        if message.get("tool_calls"):
            tools_summary = handle_tool_calls(message)

            # Passa os resultados das tools como texto em nova mensagem de usuário
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Resultados das ferramentas:\n"
                        f"{tools_summary}\n\n"
                        "Com base nesses resultados, responda à pergunta original do usuário."
                    ),
                }
            )

            choice = call_model(messages)  # 2ª chamada, sem tools, sem streaming
            message = choice["message"]

        assistant_content = (message.get("content") or "").strip()
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
            print(f"Assistente: {assistant_content}")


if __name__ == "__main__":
    chat_with_tools()
