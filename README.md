
## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Ollama / Mistral
This project uses Ollama with the Mistral model for local generation.

ollama pull mistral
Run retrieval test
python notebooks/01_retrieval_elifs_part.py
Run Ollama test
python notebooks/02_ollama_generation_test.py
Run full RAG pipeline
python notebooks/03_rag_pipeline_test.py
Run evaluation
python notebooks/05_evaluation_test.py
Run Streamlit app
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false streamlit run app.py --server.fileWatcherType none

```markdown
## Dataset Note

The repository includes a 1000-sample bug-report corpus derived from the GitHub Bugs Prediction dataset for demonstration and evaluation purposes. The original large dataset archive is not included in this repository.
