🩺 Medical Copilot
A simple, local-first Streamlit application designed to act as a "copilot" for medical professionals. This tool leverages local Large Language Models (LLMs) via Ollama to analyze patient records, answer diagnostic questions, and provide insights based on a private, custom knowledge base of medical literature.

✨ Features
Modular UI: A clean, tab-based interface built with Streamlit for easy navigation.

Local First: All components (LLMs, Vector DB, Patient Data) run locally on your machine, ensuring data privacy.

Swappable LLMs: Easily switch between different text and vision models hosted by Ollama by changing a single line of code.

RAG System: Augment the LLM's knowledge by uploading your own medical textbooks and research papers into a local ChromaDB vector store.

Patient Management: Onboard new patients by uploading their medical records (PDFs, TXT, images).

Interactive Chat: Engage in a conversation about a selected patient, with the chat history saved and reloaded for each session.

Side-by-Side Document Viewer: View a patient's PDF report directly in the UI while chatting with the copilot.

Multimodal Analysis: Upload medical images (X-rays, MRIs) during a chat session for analysis by a vision-capable model like LLaVA.

🏗️ Project Architecture
The application is built on a simple, modular architecture that separates the UI from the backend logic and data stores.
<img width="3840" height="3516" alt="flowchart" src="https://github.com/user-attachments/assets/4aa3b650-d9a6-420c-b52a-82777463da90" />



🛠️ Technology Stack
Application Framework: Streamlit

LLM Hosting: Ollama

Orchestration: LangChain

Vector Database: ChromaDB

PDF/Image Processing: PyPDF, Pillow, PyMuPDF

Language: Python 3.12+

📁 Directory Structure
Medical Co-pilot/
│
├── data/
│   ├── patient_files/      # Stores all uploaded patient records
│   └── patients.json         # Tracks patients and their chat histories
│
├── vector_db/
│   └── (ChromaDB persistent storage)
│
├── app.py                  # Main Streamlit application file (UI)
├── utils.py                # Backend logic and helper functions
├── requirements.txt        # Python dependencies
└── README.md               # You are here

🚀 Setup and Installation
Clone the Repository:

git clone <your-repository-url>
cd "Medical Co-pilot"

Create a Virtual Environment:

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Install and Run Ollama:

Download and install Ollama from ollama.com.

Pull the required models from the command line:

ollama pull llama3
ollama pull llava
ollama pull nomic-embed-text

Ensure the Ollama application is running in the background.

Run the Streamlit Application:

streamlit run app.py

📋 Usage Workflow
The application is divided into three main modules accessible from the sidebar.

1. Upload to Knowledge Base
This module populates the RAG system's knowledge base.

<img width="3840" height="465" alt="procedure flow" src="https://github.com/user-attachments/assets/8d137b39-5fb1-4680-96f7-07587550c399" />


2. Onboard New Patient
This module adds a new patient to the system.

<img width="3840" height="592" alt="patient onboard" src="https://github.com/user-attachments/assets/2d8a5c93-2019-4b35-8bf4-d8a254234302" />


3. Patient Diagnostic Chat
This is the main interaction module.

Text-Only Chat Flow
<img width="3840" height="1969" alt="text working" src="https://github.com/user-attachments/assets/b4ad0d2b-0792-4eb7-b4b5-7cbe798e8dfe" />


Image Analysis Chat Flow
<img width="3840" height="1955" alt="medical working" src="https://github.com/user-attachments/assets/09dfb488-7abe-4a66-bcd7-d57d98458259" />


