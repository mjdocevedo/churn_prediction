# Local LLM Setup Guide: Llama 3.2 with Ollama

Since our pipeline uses **llama3.2:1b** locally, follow these steps to get it running depending on your operating system environment (Windows native or Linux Virtual Machine).

---

## 1. Scenario A: Install Ollama on Windows (Native)
Ollama is the easiest way to run LLMs locally on Windows.

1.  **Download**: Go to [ollama.com/download](https://ollama.com/download) and download the Windows installer.
2.  **Install**: Run `OllamaSetup.exe` and follow the prompts.
3.  **Restart Terminal**: Once installed, you **must close and restart your terminal** (CMD or PowerShell) for the `ollama` command to be recognized.
4.  **Verify**: Open a **new** terminal and type:
    ```cmd
    ollama --version
    ```

### 🚨 Troubleshooting: Command not recognized?
If Windows still says `'ollama' is not recognized`:
1.  **Test the direct path**:
    ```cmd
    "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" --version
    ```
2.  **Fix the PATH** (Permanent fix): 
    - Press the **Windows Key**, type "Environment Variables", and select "Edit the system environment variables".
    - Click "Environment Variables" -> Under "User variables", find **Path** -> **Edit**.
    - Click **New** and paste: `%LOCALAPPDATA%\Programs\Ollama`
    - Restart your terminal.

---

## 2. Scenario B: Install Ollama on Linux (Virtual Machine)
If you are running the project inside a Linux Virtual Machine (e.g., WSL2, Ubuntu VM):

1.  **Install via curl**:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
2.  **Start the Ollama Server** (if it doesn't start automatically):
    ```bash
    sudo systemctl start ollama
    ```
3.  **Verify**:
    ```bash
    ollama --version
    ```

> **Note for VM Users**: If your Python agent runs on Windows but Ollama is inside the VM, you must configure Ollama to listen on all interfaces (e.g., set `OLLAMA_HOST=0.0.0.0` in your systemd service) and update your Windows `.env` to point to the VM's IP address (`OLLAMA_BASE_URL=http://<VM_IP>:11434`).

---

## 3. Pull and Test the Model (Both Scenarios)
Once Ollama is installed and running on your active environment:

1.  **Pull the Model**:
    ```bash
    ollama pull llama3.2:1b
    ```
2.  **Test the Model**:
    Verify it works by chatting with it directly in your terminal:
    ```bash
    ollama run llama3.2:1b "Hello, who are you?"
    ```
    *Exit the chat by typing `/bye` or hit Ctrl+D.*

---

## 4. Configure the Retention Assistant
To use this local model, update your `.env` file before running the MLFlow agent traces:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 💡 Why Llama 3.2 1B?
- **Speed**: It is extremely fast and perfect for iterative MLOps prototyping.
- **Memory**: It uses roughly 1-2GB of RAM, meaning it runs smoothly on almost any laptop without a dedicated GPU.
- **Privacy**: No data leaves your machine/VM.
- **LLMOps**: Even though the model is local, **MLflow will still trace everything** (latency, spans, prompts) exactly like it does for OpenAI endpoints!
