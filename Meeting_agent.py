import streamlit as st
import os
import tempfile
import pandas as pd
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from crewai import Agent, Task, Crew
from crewai.process import Process

# Import utility classes
from Meeting_agent.utils import TodoistTools, TranscriptExtractor, TelegramCommunicator, TaskExtractor, TodoistMeetingManager

# Helper functions
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "setup": None,
        "openai_api_key": "",
        "prepared": False,
        "vectorstore": None,
        "context_analysis": None,
        "meeting_strategy": None,
        "executive_brief": None,
        "todoist_api_key": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "transcript_source": "google_meet",
        "meeting_id": "",
        "todoist_manager": None,
        "task_extraction_results": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def process_documents(base_context, uploaded_files):
    """Process base context and uploaded documents"""
    docs = []
    
    # Add base context as document
    with tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".txt") as temp:
        temp.write(base_context)
        temp.flush()
        docs.extend(TextLoader(temp.name).load())
    
    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            suffix = file.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                tmp.write(file.getbuffer())
                tmp.flush()
                try:
                    loader = PyPDFLoader(tmp.name) if suffix == 'pdf' else TextLoader(tmp.name)
                    docs.extend(loader.load())
                    st.success(f"Processed: {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
    
    return docs

def create_vectorstore(docs):
    """Create a vector store from documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(api_key=st.session_state["openai_api_key"])
    return FAISS.from_documents(splits, embeddings)

def run_crewai_analysis(setup, llm):
    """Run CrewAI analysis for meeting preparation"""
    attendees_text = "\n".join([f"- {attendee}" for attendee in setup['attendees']])
    
    # Create agents
    context_agent = Agent(
        role='Context Analyst',
        goal='Provide comprehensive context analysis for the meeting',
        backstory="""You are an expert business analyst who specializes in preparing context documents for meetings. 
        You thoroughly research companies and identify key stakeholders.""",
        llm=llm,
        verbose=True
    )
    
    strategy_agent = Agent(
        role='Meeting Strategist',
        goal='Create detailed meeting strategy and agenda',
        backstory="""You are a seasoned meeting facilitator who excels at structuring effective business discussions.
        You understand how to allocate time optimally.""",
        llm=llm,
        verbose=True
    )
    
    brief_agent = Agent(
        role='Executive Briefer',
        goal='Generate executive briefing with actionable insights',
        backstory="""You are a master communicator who specializes in crafting executive briefings.
        You distill complex information into clear, concise documents.""",
        llm=llm,
        verbose=True
    )
    
    # Create tasks
    context_task = Task(
        description=f"""Analyze the context for the meeting with {setup['company']}.
Consider:
1. Company background and market position
2. Meeting objective: {setup['objective']}
3. Attendees: {attendees_text}
4. Focus areas: {setup['focus']}

FORMAT IN MARKDOWN with clear headings.
""",
        agent=context_agent,
        expected_output="""A markdown-formatted context analysis with sections for Executive Summary, 
        Company Background, Situation Analysis, Key Stakeholders, and Strategic Considerations."""
    )
    
    strategy_task = Task(
        description=f"""Develop a meeting strategy for the {setup['duration']}-minute meeting with {setup['company']}.
Include:
1. Time-boxed agenda with specific allocations
2. Key talking points for each section
3. Discussion questions and role assignments

FORMAT IN MARKDOWN with clear headings.
""",
        agent=strategy_agent,
        expected_output="""A markdown-formatted meeting strategy with sections for Meeting Overview, 
        Detailed Agenda, Key Talking Points, and Success Criteria."""
    )
    
    brief_task = Task(
        description=f"""Create an executive briefing for the meeting with {setup['company']}.
Include:
1. Executive summary with key points
2. Key talking points and recommendations
3. Anticipated questions and prepared answers

FORMAT IN MARKDOWN with clear headings.
""",
        agent=brief_agent,
        expected_output="""A markdown-formatted executive briefing with sections for Executive Summary, 
        Key Talking Points, Q&A Preparation, and Next Steps."""
    )
    
    # Run crew
    crew = Crew(
        agents=[context_agent, strategy_agent, brief_agent],
        tasks=[context_task, strategy_task, brief_task],
        verbose=True,
        process=Process.sequential
    )
    
    # Execute crew
    return crew.kickoff()

def extract_content(result_item):
    """Extract content from CrewAI result item"""
    if hasattr(result_item, 'result'):
        return result_item.result
    if isinstance(result_item, dict) and 'result' in result_item:
        return result_item['result']
    if isinstance(result_item, str):
        return result_item
    return str(result_item)

def fallback_analysis(setup, llm):
    """Fallback method if CrewAI fails"""
    attendees_text = "\n".join([f"- {attendee}" for attendee in setup['attendees']])
    
    context_prompt = f"""Analyze the context for the meeting with {setup['company']}:
    - Meeting objective: {setup['objective']}
    - Attendees: {attendees_text}
    - Focus areas: {setup['focus']}
    
    Format in markdown with appropriate headings."""
    
    strategy_prompt = f"""Create a meeting strategy for the {setup['duration']}-minute meeting with {setup['company']}:
    - Meeting objective: {setup['objective']}
    - Focus areas: {setup['focus']}
    
    Format in markdown with appropriate headings."""
    
    brief_prompt = f"""Create an executive brief for the meeting with {setup['company']}:
    - Meeting objective: {setup['objective']}
    - Focus areas: {setup['focus']}
    
    Format in markdown with appropriate headings."""
    
    context_content = llm.invoke(context_prompt).content
    strategy_content = llm.invoke(strategy_prompt).content
    brief_content = llm.invoke(brief_prompt).content
    
    return context_content, strategy_content, brief_content

def create_qa_chain(vectorstore, api_key):
    """Create a QA chain for answering questions"""
    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following context to answer the question.
        If you don't know the answer, say that you don't know.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create QA chain
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0.7, api_key=api_key),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

def process_transcript(transcript, todoist_manager):
    """Process a transcript and extract tasks"""
    # Create task extractor
    task_extractor = TaskExtractor(todoist_manager.task_extractor.llm)
    
    # Extract tasks from transcript
    extracted_data = task_extractor.extract_tasks_from_transcript(transcript)
    
    # Check for extraction errors
    if "error" in extracted_data:
        return {"error": extracted_data["error"]}
    
    # Create projects and tasks
    results = {
        "projects_created": [],
        "tasks_created": []
    }
    
    for project_data in extracted_data["projects"]:
        project_name = project_data["name"]
        project = todoist_manager.todoist_tools.create_project(project_name)
        
        if "error" in project:
            results["error"] = project["error"]
            return results
        
        results["projects_created"].append(project_name)
        
        for task_data in project_data["tasks"]:
            task = todoist_manager.todoist_tools.create_and_assign_task(
                task_data["content"],
                project_name,
                task_data.get("assignee"),
                task_data.get("due_string"),
                task_data.get("priority", 3)
            )
            
            if "error" in task:
                results["task_errors"] = results.get("task_errors", []) + [task["error"]]
            else:
                results["tasks_created"].append({
                    "content": task_data["content"],
                    "project": project_name,
                    "assignee": task_data.get("assignee")
                })
    
    return results

def send_telegram_notification(telegram_bot_token, telegram_chat_id, results):
    """Send notification about tasks to Telegram"""
    # Initialize Telegram communicator
    telegram = TelegramCommunicator(telegram_bot_token, telegram_chat_id)
    
    # Create summary message
    summary = f"*Meeting Task Summary*\n\n"
    summary += f"Projects: {', '.join(results['projects_created'])}\n\n"
    summary += f"Tasks Created: {len(results['tasks_created'])}\n\n"
    
    for task in results["tasks_created"]:
        summary += f"- {task['content']} (Project: {task['project']}"
        if task.get("assignee"):
            summary += f", Assigned to: {task['assignee']}"
        summary += ")\n"
    
    # Send message
    return telegram.send_message(summary)

# Main app
def main():
    # Page setup
    st.set_page_config(page_title="AI Meeting Assistant", page_icon="📝", layout="wide")
    st.title("📝 AI Meeting Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for API keys
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state["openai_api_key"])
        if openai_api_key:
            st.session_state["openai_api_key"] = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Add Todoist API key input
        todoist_api_key = st.text_input("Todoist API Key", type="password", value=st.session_state["todoist_api_key"])
        if todoist_api_key != st.session_state["todoist_api_key"]:
            st.session_state["todoist_api_key"] = todoist_api_key
            # Reset todoist manager when API key changes
            st.session_state["todoist_manager"] = None
        
        # Add Telegram API key inputs (optional)
        with st.expander("Telegram Integration (Optional)"):
            telegram_bot_token = st.text_input("Telegram Bot Token", type="password", value=st.session_state["telegram_bot_token"])
            telegram_chat_id = st.text_input("Telegram Chat ID", value=st.session_state["telegram_chat_id"])
            
            # Check if credentials changed
            telegram_credentials_changed = False
            if telegram_bot_token != st.session_state["telegram_bot_token"]:
                st.session_state["telegram_bot_token"] = telegram_bot_token
                telegram_credentials_changed = True

            if telegram_chat_id != st.session_state["telegram_chat_id"]:
                st.session_state["telegram_chat_id"] = telegram_chat_id
                telegram_credentials_changed = True

            # Reinitialize manager if credentials changed
            if telegram_credentials_changed and st.session_state["todoist_api_key"]:
                st.session_state["todoist_manager"] = None  # Force reinitialization
        
        st.info("This app helps prepare for meetings by analyzing company info, creating agendas, answering questions, and managing tasks.")
    
    # Main tabs
    tab_setup, tab_results, tab_qa, tab_tasks = st.tabs(["Meeting Setup", "Preparation Results", "Q&A Assistant", "Task Management"])

    # Meeting Setup Tab
    with tab_setup:
        st.subheader("Meeting Configuration")
        company_name = st.text_input("Company Name")
        meeting_objective = st.text_area("Meeting Objective")
        meeting_date = st.date_input("Meeting Date")
        meeting_duration = st.slider("Meeting Duration (minutes)", 15, 180, 60)

        st.subheader("Attendees")
        attendees_data = st.data_editor(
            pd.DataFrame({"Name": [""], "Role": [""], "Company": [""]}), 
            num_rows="dynamic",
            use_container_width=True
        )
        focus_areas = st.text_area("Focus Areas or Concerns")
        
        uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["txt", "pdf"])

        if st.button("Prepare Meeting", type="primary", use_container_width=True):
            if not openai_api_key or not company_name or not meeting_objective:
                st.error("Please fill in all required fields and API key.")
            else:
                attendees_formatted = []
                for _, row in attendees_data.iterrows():
                    if row["Name"]:  # Skip empty rows
                        attendees_formatted.append(f"{row['Name']}, {row['Role']}, {row['Company']}")
                
                st.session_state["setup"] = {
                    "company": company_name,
                    "objective": meeting_objective,
                    "date": meeting_date,
                    "duration": meeting_duration,
                    "attendees": attendees_formatted,
                    "focus": focus_areas,
                    "files": uploaded_files
                }
                st.session_state["prepared"] = False
                st.rerun()

    # Preparation Results Tab
    with tab_results:
        if st.session_state["setup"] and not st.session_state["prepared"]:
            with st.status("Processing meeting data...", expanded=True) as status:
                setup = st.session_state["setup"]
                
                # Create base context
                attendees_text = "\n".join([f"- {attendee}" for attendee in setup['attendees']])
                base_context = f"""
Meeting Information:
- Company: {setup['company']}
- Objective: {setup['objective']}
- Date: {setup['date']}
- Duration: {setup['duration']} minutes
- Focus Areas: {setup['focus']}

Attendees:
{attendees_text}
"""
                # Process documents
                docs = process_documents(base_context, setup['files'])
                
                # Create vector store
                vectorstore = create_vectorstore(docs)
                st.session_state["vectorstore"] = vectorstore
                
                # Initialize LLM
                llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=st.session_state["openai_api_key"])
                
                # Try CrewAI approach first
                try:
                    result = run_crewai_analysis(setup, llm)
                    
                    if isinstance(result, list) and len(result) >= 3:
                        context_content = extract_content(result[0])
                        strategy_content = extract_content(result[1])
                        brief_content = extract_content(result[2])
                    else:
                        raise Exception("CrewAI did not return expected format")
                    
                except Exception as e:
                    st.warning(f"Using fallback method. Error: {str(e)}")
                    context_content, strategy_content, brief_content = fallback_analysis(setup, llm)
                
                # Store results
                st.session_state.update({
                    "context_analysis": context_content,
                    "meeting_strategy": strategy_content,
                    "executive_brief": brief_content,
                    "prepared": True
                })
                
                status.update(label="Meeting preparation complete!", state="complete", expanded=False)

        if st.session_state["prepared"]:
            # Show results tabs
            results_tab1, results_tab2, results_tab3 = st.tabs(["Context Analysis", "Meeting Strategy", "Executive Brief"])
            
            with results_tab1:
                if st.session_state["context_analysis"]:
                    st.markdown(st.session_state["context_analysis"])
                else:
                    st.warning("Context analysis not generated")
            
            with results_tab2:
                if st.session_state["meeting_strategy"]:
                    st.markdown(st.session_state["meeting_strategy"])
                else:
                    st.warning("Meeting strategy not generated")
            
            with results_tab3:
                if st.session_state["executive_brief"]:
                    st.markdown(st.session_state["executive_brief"])
                else:
                    st.warning("Executive brief not generated")
            
            # Download buttons

            # Download buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.session_state["context_analysis"]:
                    st.download_button("Download Context Analysis", st.session_state["context_analysis"], 
                                    "context_analysis.md", use_container_width=True)
            with col2:
                if st.session_state["meeting_strategy"]:
                    st.download_button("Download Meeting Strategy", st.session_state["meeting_strategy"], 
                                    "meeting_strategy.md", use_container_width=True)
            with col3:
                if st.session_state["executive_brief"]:
                    st.download_button("Download Executive Brief", st.session_state["executive_brief"], 
                                    "executive_brief.md", use_container_width=True)
        else:
            st.info("Please configure your meeting in the 'Meeting Setup' tab.")

    # Q&A Assistant Tab
    with tab_qa:
        st.subheader("Meeting Q&A Assistant")
        
        if not st.session_state["openai_api_key"]:
            st.warning("Please enter your OpenAI API key in the sidebar.")
        elif st.session_state["vectorstore"] is None:
            st.info("Please prepare a meeting first to use the Q&A feature.")
        else:
            st.success("Ask questions about your meeting below:")
            
            # Direct Q&A using the vectorstore
            question = st.text_input("Your question:", key="qa_question")
            
            if question:
                with st.spinner("Finding answer..."):
                    try:
                        # Create QA chain
                        qa = create_qa_chain(st.session_state["vectorstore"], st.session_state["openai_api_key"])
                        
                        # Run query and get result
                        result = qa.invoke({"query": question})
                        
                        # Display answer
                        st.markdown("### Answer")
                        st.markdown(result["result"])
                        
                        # Show sources
                        with st.expander("View Source Documents"):
                            for i, doc in enumerate(result.get("source_documents", [])):
                                st.markdown(f"**Source {i+1}**")
                                st.markdown(f"```\n{doc.page_content}\n```")
                                st.divider()
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.error("Please check your question and try again.")

    # Task Management Tab
    with tab_tasks:
        st.subheader("Meeting Task Management")
        
        if not st.session_state["openai_api_key"]:
            st.warning("Please enter your OpenAI API key in the sidebar.")
        elif not st.session_state["todoist_api_key"]:
            st.warning("Please enter your Todoist API key in the sidebar to use task management features.")
        else:
            # Initialize Todoist Meeting Manager if not already done
            if st.session_state["todoist_manager"] is None:
                llm = ChatOpenAI(model="gpt-4", temperature=0.3, api_key=st.session_state["openai_api_key"])
                st.session_state["todoist_manager"] = TodoistMeetingManager(
                    st.session_state["todoist_api_key"],
                    st.session_state["telegram_bot_token"] if st.session_state["telegram_bot_token"] else None,
                    st.session_state["telegram_chat_id"] if st.session_state["telegram_chat_id"] else None,
                    st.session_state["transcript_source"],
                    llm
                )
            
            # Transcript source selection
            col1, col2 = st.columns([1, 2])
            with col1:
                transcript_source = st.selectbox(
                    "Transcript Source",
                    ["google_meet", "whatsapp", "telegram"],
                    index=["google_meet", "whatsapp", "telegram"].index(st.session_state["transcript_source"])
                )
                if transcript_source != st.session_state["transcript_source"]:
                    st.session_state["transcript_source"] = transcript_source
                    # Update transcript extractor in the manager
                    st.session_state["todoist_manager"].transcript_extractor = TranscriptExtractor(transcript_source)
            
            with col2:
                meeting_id = st.text_input(
                    "Meeting/Chat ID", 
                    value=st.session_state["meeting_id"],
                    help="Enter the ID of your Google Meet, WhatsApp, or Telegram conversation"
                )
                if meeting_id != st.session_state["meeting_id"]:
                    st.session_state["meeting_id"] = meeting_id
            
            # Option to manually input transcript
            with st.expander("Manual Transcript Input (Optional)"):
                manual_transcript = st.text_area(
                    "Enter Meeting Transcript", 
                    height=200,
                    help="If you don't have API access, you can manually paste a transcript here"
                )
            
            # Process transcript and extract tasks
            if st.button("Extract Tasks from Meeting", type="primary", use_container_width=True):
                if not meeting_id and not manual_transcript:
                    st.error("Please provide either a Meeting ID or a manual transcript.")
                else:
                    with st.spinner("Processing meeting transcript and extracting tasks..."):
                        try:
                            # If manual transcript is provided, use that instead of API
                            if manual_transcript:
                                results = process_transcript(manual_transcript, st.session_state["todoist_manager"])
                            else:
                                # Use the manager to process the meeting
                                results = st.session_state["todoist_manager"].process_meeting(meeting_id)
                            
                            st.session_state["task_extraction_results"] = results
                            
                        except Exception as e:
                            st.error(f"Error processing meeting: {str(e)}")
            
            # Display task extraction results
            if st.session_state["task_extraction_results"]:
                results = st.session_state["task_extraction_results"]
                
                if "error" in results:
                    st.error(f"Error: {results['error']}")
                else:
                    st.success("Meeting processed successfully!")
                    
                    # Display projects created
                    if results["projects_created"]:
                        st.subheader("Projects Created")
                        for project in results["projects_created"]:
                            st.write(f"- {project}")
                    
                    # Display tasks created
                    if results["tasks_created"]:
                        st.subheader("Tasks Created")
                        task_df = pd.DataFrame(results["tasks_created"])
                        st.dataframe(task_df)
                    
                    # Display task errors if any
                    if "task_errors" in results and results["task_errors"]:
                        with st.expander("Task Creation Errors"):
                            for error in results["task_errors"]:
                                st.error(error)
                    
                    # Option to send notification to team
                    if st.session_state["telegram_bot_token"] and st.session_state["telegram_chat_id"]:
                        if st.button("Notify Team on Telegram"):
                            with st.spinner("Sending notification..."):
                                try:
                                    message_result = send_telegram_notification(
                                        st.session_state["telegram_bot_token"],
                                        st.session_state["telegram_chat_id"],
                                        results
                                    )
                                    
                                    if "error" in message_result:
                                        st.error(f"Error sending notification: {message_result['error']}")
                                    else:
                                        st.success("Team notification sent successfully!")
                                except Exception as e:
                                    st.error(f"Error sending notification: {str(e)}")
            
            # Project and task management section
            with st.expander("Manage Todoist Projects and Tasks"):
                if st.button("Refresh Projects"):
                    try:
                        projects = st.session_state["todoist_manager"].todoist_tools.get_projects()
                        if "error" in projects:
                            st.error(f"Error fetching projects: {projects['error']}")
                        else:
                            # Display projects in a dataframe
                            project_df = pd.DataFrame([{"id": p["id"], "name": p["name"]} for p in projects])
                            st.dataframe(project_df)
                    except Exception as e:
                        st.error(f"Error fetching projects: {str(e)}")
                
                # Form to create a new task
                st.subheader("Create New Task")
                with st.form("new_task_form"):
                    task_content = st.text_input("Task Description")
                    
                    # Project selection
                    try:
                        projects = st.session_state["todoist_manager"].todoist_tools.get_projects()
                        if "error" not in projects:
                            project_names = [p["name"] for p in projects]
                            selected_project = st.selectbox("Project", project_names)
                        else:
                            selected_project = st.text_input("Project Name (could not fetch existing projects)")
                    except:
                        selected_project = st.text_input("Project Name")
                    
                    assignee = st.text_input("Assignee (if applicable)")
                    due_string = st.text_input("Due Date (e.g., 'tomorrow', 'next Monday')")
                    priority = st.slider("Priority", 1, 4, 3, help="4 is highest priority")
                    
                    submit_task = st.form_submit_button("Create Task")
                    
                    if submit_task:
                        if not task_content or not selected_project:
                            st.error("Task description and project are required.")
                        else:
                            try:
                                task = st.session_state["todoist_manager"].todoist_tools.create_and_assign_task(
                                    task_content,
                                    selected_project,
                                    assignee if assignee else None,
                                    due_string if due_string else None,
                                    priority
                                )
                                
                                if "error" in task:
                                    st.error(f"Error creating task: {task['error']}")
                                else:
                                    st.success(f"Task '{task_content}' created successfully!")
                            except Exception as e:
                                st.error(f"Error creating task: {str(e)}")
                                
            # Connect meeting with tasks
            st.subheader("Connect Meeting to Tasks")
            
            if st.session_state["prepared"] and st.session_state["todoist_api_key"]:
                if st.button("Generate Tasks from Meeting Preparation"):
                    try:
                        with st.spinner("Analyzing meeting materials and creating tasks..."):
                            # Combine all meeting materials
                            meeting_content = f"""
                            Meeting Context:
                            {st.session_state.get('context_analysis', '')}
                            
                            Meeting Strategy:
                            {st.session_state.get('meeting_strategy', '')}
                            
                            Executive Brief:
                            {st.session_state.get('executive_brief', '')}
                            """
                            
                            # Process the combined meeting content
                            results = process_transcript(meeting_content, st.session_state["todoist_manager"])
                            st.session_state["task_extraction_results"] = results
                            
                            if "error" in results:
                                st.error(f"Error extracting tasks: {results['error']}")
                            elif results["tasks_created"]:
                                st.success(f"Created {len(results['tasks_created'])} tasks in Todoist based on meeting preparation!")
                                
                                # Display tasks in a table
                                task_df = pd.DataFrame(results["tasks_created"])
                                st.dataframe(task_df)
                            else:
                                st.warning("No tasks were identified in the meeting materials.")
                    
                    except Exception as e:
                        st.error(f"Error creating tasks from meeting: {str(e)}")
            else:
                if not st.session_state["prepared"]:
                    st.info("Please prepare a meeting first in the 'Meeting Setup' tab.")
                if not st.session_state["todoist_api_key"]:
                    st.info("Please enter your Todoist API key in the sidebar.")

# Run the app
if __name__ == "__main__":
    main()