📝 AI Meeting Assistant
An AI-powered Meeting Preparation and Task Management Assistant
Built with LangChain, CrewAI, OpenAI, Todoist, Telegram, and Streamlit.

📌 Project Overview
This application helps you prepare for business meetings intelligently.
It analyzes company and meeting context, creates agendas, summarizes meetings, extracts action tasks, and notifies teams via Telegram – all powered by AI.

Key Features:

📚 Prepare context analysis, meeting strategies, and executive briefs using CrewAI agents.

🤖 Use LLMs (GPT-4 via OpenAI) for fallback analysis if needed.

📂 Upload PDFs or text files as supporting documents.

💬 Interact with a Q&A bot trained on meeting materials.

📋 Extract tasks from meeting transcripts or materials.

✅ Assign tasks and create projects directly in Todoist.

📲 Send task summaries and notifications directly to Telegram groups or users.

🛠 Tech Stack

Technology	Purpose
Streamlit	Web Application UI
OpenAI (GPT-4)	LLM for context generation and Q&A
CrewAI	Multi-agent collaborative processing
LangChain	Vector storage (FAISS), document splitting
Todoist API	Task and project management
Telegram Bot API	Notifications for meeting tasks
FAISS	Vector database for question answering
🚀 Setup Instructions
1. Clone this Repository
bash
Copy
Edit
git clone https://github.com/your-username/ai-meeting-assistant.git
cd ai-meeting-assistant
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
Dependencies include:

streamlit

langchain

crewai

openai

faiss-cpu

pandas

python-dotenv

todoist-python

python-telegram-bot

etc.

(You can generate requirements.txt easily if you need.)

3. API Keys Required
You will need the following:

🔑 OpenAI API Key (for GPT-4 usage)

📝 Todoist API Key (for managing tasks)

📣 Telegram Bot Token + Chat ID (optional, for team notifications)

You can enter these in the sidebar when the app starts.

4. Run the App
bash
Copy
Edit
streamlit run app.py
📋 Core Features Explained
➡️ Meeting Setup
Configure company, meeting objective, attendees, duration, and focus areas.

Upload PDFs or text files for additional meeting context.

➡️ Preparation Results
Agents prepare:

📄 Context Analysis (company background, stakeholders)

🗂️ Meeting Strategy (agenda, questions, role assignments)

🧠 Executive Brief (summary, key points, Q&A prep)

Download all generated documents in markdown format.

➡️ Q&A Assistant
Ask questions about the meeting using a ChatGPT-powered assistant.

Sources from your uploaded files and context documents.

➡️ Task Management
Extract action tasks from meeting transcript or preparation.

Create Todoist projects automatically.

Assign tasks to teammates.

Send task summaries to your Telegram group or DM.

🧠 How Agents Work (CrewAI)
Agents created:

🧠 Context Analyst: Prepares detailed background analysis.

🎯 Meeting Strategist: Designs agenda and success criteria.

🗣️ Executive Briefer: Crafts summarized executive briefing.

If CrewAI fails for any reason, a fallback GPT-4 prompt method is used automatically.

🌐 Integration Overview
Todoist:

Create and manage projects and tasks.

Automatically assign tasks with deadlines and priorities.

Telegram:

Send meeting summaries and task lists to a team via a bot.

Google Meet, WhatsApp, Telegram transcripts:

Parse meeting transcripts to generate tasks automatically.

📦 Project Structure
graphql
Copy
Edit
├── app.py               # Streamlit app (Main file)
├── utils/
│   ├── TodoistTools.py   # Manage Todoist API integration
│   ├── TranscriptExtractor.py # Extract transcript text
│   ├── TelegramCommunicator.py # Send Telegram messages
│   ├── TaskExtractor.py  # Extract tasks from transcripts
│   └── TodoistMeetingManager.py # Meeting management orchestration
├── requirements.txt
└── README.md