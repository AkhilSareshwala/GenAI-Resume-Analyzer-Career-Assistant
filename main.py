from flask import Flask, request, render_template, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
import PyPDF2
import uuid
import logging
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Verify Google API key is available
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,
    convert_system_message_to_human=True
)

# Resume analysis prompt template
RESUME_SUMMARY_TEMPLATE = """
As an AI Career Coach, analyze this resume and provide a structured summary:

1. Career Profile:
   - Key professional identity
   - Years of experience
   - Industry focus

2. Core Competencies:
   - Technical skills (list top 5-8)
   - Soft skills (list top 3-5)

3. Professional Experience:
   - For each role:
     * Position & Company
     * Duration
     * Key achievements (quantify where possible)

4. Education & Certifications:
   - Degrees with institutions
   - Relevant certifications

5. Notable Achievements:
   - Awards
   - Publications
   - Significant projects

Format the output with clear section headings and bullet points for readability.

Resume Content:
{resume}
"""

resume_prompt = PromptTemplate.from_template(RESUME_SUMMARY_TEMPLATE)
resume_analyzer = resume_prompt | llm

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with error handling"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join(page.extract_text() for page in reader.pages)
            if not text.strip():
                raise ValueError("The PDF appears to be empty or contains no text")
            return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

def create_vectorstore(text_chunks):
    """Create FAISS vectorstore with HuggingFace embeddings"""
    try:
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        raise RuntimeError(f"Vectorstore creation failed: {str(e)}")

def get_conversation_chain(vectorstore):
    """Create conversational chain with memory"""
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        get_chat_history=lambda h: h,
        memory=memory,
        return_source_documents=True
    )

@app.route('/')
def index():
    """Render upload page"""
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process resume
        resume_text = extract_text_from_pdf(file_path)
        text_chunks = text_splitter.split_text(resume_text)
        
        # Store in session for chat functionality
        session['resume_text'] = resume_text
        
        # Create and save vectorstore
        vectorstore = create_vectorstore(text_chunks)
        vectorstore.save_local("vector_index")
        
        # Generate analysis
        analysis = resume_analyzer.invoke({"resume": resume_text})
        
        return render_template('results.html', 
                           resume_analysis=analysis.content,
                           filename=filename)
    
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/ask', methods=['GET', 'POST'])
def ask_query():
    """Handle Q&A about the resume"""
    if request.method == 'POST':
        try:
            query = request.form['query']
            db = FAISS.load_local(
                "vector_index", 
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = db.as_retriever(search_kwargs={"k": 4})
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            result = qa_chain.invoke({"query": query})
            return render_template('qa_results.html', 
                               query=query,
                               result=result['result'])
        
        except Exception as e:
            return render_template('error.html', error=str(e)), 500
    
    return render_template('ask.html')

@app.route('/chat_resume', methods=['GET', 'POST'])
def chat_resume():
    # Initialize chat if not exists
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
    
    try:
        # Always load fresh vectorstore
        db = FAISS.load_local(
            "vector_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Initialize memory if not exists
        if 'memory' not in session:
            session['memory'] = {
                'chat_history': []
            }
            
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Restore previous messages
        for msg in session['memory']['chat_history']:
            if msg['type'] == 'human':
                memory.chat_memory.add_user_message(msg['content'])
            else:
                memory.chat_memory.add_ai_message(msg['content'])
        
        # Create fresh chain each time
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True,
            get_chat_history=lambda h: h
        )

        if request.method == 'POST':
            user_question = request.form['user_question']
            response = conversation_chain({"question": user_question})
            
            # Update session memory
            session['memory']['chat_history'].extend([
                {"type": "human", "content": user_question},
                {"type": "ai", "content": response['answer']}
            ])
            session.modified = True
            
            # Format history for display
            formatted_history = []
            for msg in session['memory']['chat_history']:
                role = "You" if msg['type'] == 'human' else "Career Coach"
                formatted_history.append((role, msg['content']))
            
            return render_template('resume_chat.html', 
                               chat_history=formatted_history)
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return render_template('error.html', error=str(e)), 500
    
    # For GET requests or fallthrough
    formatted_history = [
        (role, content) for msg in session.get('memory', {}).get('chat_history', [])
        for role, content in [("You" if msg['type'] == 'human' else "Career Coach", msg['content'])]
    ]
    return render_template('resume_chat.html', 
                       chat_history=formatted_history)
