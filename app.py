import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import json
 
# Sidebar contents
with st.sidebar:
    st.title('💬 PDFBot Plus 📚')
    st.markdown('''
    Esta aplicación es un chatbot impulsado por LLM construido usando:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)
 
    ''')
    add_vertical_space(5)
    st.write('Desarrollado por [Mateo Heras](https://mateoheras77.github.io/WEB/)')
    st.write('Correo: wmateohv@hotmail.com')
    st.write('LinkedIn: [Mateo Heras](https://www.linkedin.com/in/mateoheras/)')

 
load_dotenv() #Cargar las variables de entorno



def main():
    
    # Verificar si la variable de sesión 'historial' ya existe, si no, inicializarla como una lista vacía
    if 'historial' not in st.session_state:
        st.session_state.historial = []

    st.header("💬 PDFBot Plus: Tu Asistente para PDFs 📚")
    st.write("¡Olvida la búsqueda tediosa en documentos!")
    st.write(" Con PDFBot Plus, \
            interactuar con archivos PDF es pan comido. 🚀🔍 Nuestro chat avanzado con IA, \
            respaldado por la potencia de OpenAI, revoluciona cómo obtienes información. \
            Solo pregúntale y PDFBot extraerá datos al instante de los PDFs, ¡como magia! ✨📄")
    st.write("Disfruta de conversaciones naturales y recupera conocimiento al vuelo, todo con una interfaz intuitiva.\
              ¡Simplifica tu vida digital con PDFBot Plus! 💡📑")
 
    # upload a PDF file
    pdf = st.file_uploader("📤 ¡Sube tu PDF aquí! 📎", type='pdf')

    # Obtener el nombre del archivo sin la extensión para usarlo como nombre del archivo JSON
    if pdf is not None:
        store_name = pdf.name[:-4]
    else:
        store_name = None

    # Cargar el historial desde el archivo JSON si el PDF es el mismo
    if store_name:
        historial_filename = f"./HISTORIAL/{store_name}_historial.json"
        if os.path.exists(historial_filename):
            with open(historial_filename, "r") as f:
                st.session_state.historial = json.load(f)


    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)  PKL/ConstEcua.pkl
 
        if os.path.exists(f"./PKL/{store_name}.pkl"):
            with open(f"./PKL/{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
                st.write('(Embeddings cargados del historial)')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"./PKL/{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)        

        # Accept user questions/query
        query = st.text_input("Preguntale a tu PDF:")
        
        # st.write(query)
 


        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = ChatOpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
            
            # Agregar la consulta y la respuesta actual al historial
            st.session_state.historial.append({"consulta": query, "respuesta": response})
            # Invertir el orden del historial para mostrar las últimas preguntas primero
            st.session_state.historial = st.session_state.historial[::-1]

            # Guardar el historial en el archivo JSON
            if store_name:
                historial_filename = f"./HISTORIAL/{store_name}_historial.json"
                with open(historial_filename, "w") as f:
                    json.dump(st.session_state.historial, f)

         # Mostrar el historial usando st.session_state.historial
        if st.session_state.historial:
            st.write("---")
            st.write("💬 ¡Echa un vistazo al Historial! 📜")
            st.write("Aquí te presentamos el recorrido de consultas y respuestas: 🔄")
            st.write("---")
            for item in st.session_state.historial:
                st.write("Consulta:", item["consulta"])
                st.write("Respuesta:", item["respuesta"])
                st.write("---")
 
if __name__ == '__main__':
    main()