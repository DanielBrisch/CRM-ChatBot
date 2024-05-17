import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from openai import OpenAI 

load_dotenv()

documents = SimpleDirectoryReader('./data').load_data()

api_key = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TEMPLATE_STR = (
    "Apenas fornecemos informações do contexto abaixo.\n"
    "-------------------------------------------------\n"
    "{context_str}\n"
    "-------------------------------------------------\n"
    "Somente com base nessas informações, por favor responda a seguinte pergunta: {query_str}\n"
)

QA_TEMPLATE = PromptTemplate(TEMPLATE_STR)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE)

def reformulate_text(text):
    response = api_key.chat.completions.create(
        model="gpt-4", 
        messages=[
            {"role": "system", "content": "Você é um assistente útil que serve para reformular textos e deixar mais legiveis e atraetes para os humanos"},
            {"role": "user", "content": f"Reformule o seguinte texto: {text}"}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    reformulated_text = response.choices[0].message.content
    return reformulated_text

def run_code(question):
    if question:
        answer = query_engine.query(question)
        reformulated_answer = reformulate_text(str(answer))
        response = f"{str(reformulated_answer)}"
        return response
    else:
        return "Por favor, insira uma pergunta."

iface = gr.Interface(
    fn=run_code,
    inputs="text",
    outputs="text",
    title="Chatbot CRM",
    description="ChatBot para auxiliar no uso do CRM Web",
)

iface.launch()

