import requests
import json

URL = "http://127.0.0.1:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

messages = [
    {
        "role": "system",
        "content": (
            "Você é direto, objetivo e não divaga. "
            "Responda sempre claro, de forma curta, exceto quando o usuário pedir profundidade."
        )
    }
]

def stream_completion():
    payload = {
        "model": "Ministral-3-8B-Reasoning-2512-Q5_K_M.gguf",
        "messages": messages,
        "temperature": 0.2,
        "stream": True,
    }

    buffer = ""   # <- acumulador REAL da resposta final

    with requests.post(URL, headers=headers, json=payload, stream=True) as r:
        for raw_line in r.iter_lines():
            if not raw_line:
                continue

            line = raw_line.decode("utf-8").strip()

            if not line.startswith("data:"):
                continue

            content = line[len("data:"):].strip()

            if content == "[DONE]":
                break

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue

            delta = data["choices"][0]["delta"]

            if "content" in delta:
                token = delta["content"]
                if token:                # <-- IGNORA tokens vazios
                    print(token, end="", flush=True)
                    buffer += token      # <-- salva a resposta real

    return buffer.strip()  # <-- devolve resposta limpa


print("🟢 Chat iniciado. Digite sua pergunta. ('sair' para encerrar)\n")

while True:
    user_input = input("\nVocê: ").strip()

    if user_input.lower() in ("sair", "exit", "quit", "bye"):
        print("🔴 Chat encerrado.")
        break

    # adiciona mensagem do usuário
    messages.append({"role": "user", "content": user_input})

    print("Assistente: ", end="", flush=True)

    # STREAM + captura real
    assistant_response = stream_completion()

    # Se veio vazio, não salva nada
    if assistant_response:
        messages.append({"role": "assistant", "content": assistant_response})
