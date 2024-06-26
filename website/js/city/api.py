from flask import Flask, jsonify, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Langchain components
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# Define API routes
@app.route('/process_text', methods=['POST', 'GET'])
def process_text():
    # Extract text from request
    data = request.json
    text = data['text']
    #text = 'Its too hot in Delhi, where can I go to get some relief?'
    
    # Split text into documents
    #docs = text_splitter.split_documents(text)
    
    # Embed documents
    #embeddings_results = embeddings.embed_documents(docs)
    
    # Process embeddings_results as needed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    search_results = new_db.similarity_search_with_relevance_scores(text, k = 30)



    #final_results = search_results[0:10]
    
    # Return response
    final_results = []
    for doc_tuple in search_results[0:10]:
        doc = doc_tuple[0]  # Access the Document object in the tuple
        doc_data = {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }
        final_results.append(doc_data)

    # Return response
    return jsonify({'results': final_results})

# Run Flask app
if __name__ == '__main__':
    app.run(port=9001, host='0.0.0.0', debug=True)
