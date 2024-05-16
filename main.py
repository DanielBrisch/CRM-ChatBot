import os 
import gradio as gr 
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

documents = SimpleDirectoryReader('./data').load_data()

TEMPLATE_STR = (
    "Apenas fornecemos informaccoes do contexto abaixo.\n"
    "-------------------------------------------------"
    "{context_str}"
    "\n-------------------------------------------------\n"
    "Somente com base nessas informacoes, por favor responda a seguinte pergunta: {query_str}\n"
)

QA_TEMPLATE = PromptTemplate(TEMPLATE_STR)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE)

def run_code(question):
    if question:
        answer = query_engine.query(question)
        sources = answer.source_nodes
        response = ""
        
        response += "\n\nResposta á pergunta:\n" + str(answer) + "\n Fonte:" 
        return response
    else: 
        return "Por favor, insira uma pergunta."

iface = gr.Interface(
    fn=run_code,
    inputs="text",
    outputs="text",
    title="Chatbot CRM",
    description="chatBot para auxiliar no uso do CRM Web",
)
    


iface.launch()
