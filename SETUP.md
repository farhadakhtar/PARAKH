# Local Setup

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Node.js 18+
- Ollama with the `llama3.1:8b` model pulled
- Git

## Setup Commands

Clone the repository and enter the project directory:

```bash
git clone https://github.com/farhadakhtar/PARAKH.git
cd PARAKH
```

Install backend dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
npm install
```

Start Docker services:

```bash
docker compose up -d
```

Pull and run Ollama locally:

```bash
ollama pull llama3.1:8b
ollama serve
```

In another terminal, verify the model is available:

```bash
ollama list
```
