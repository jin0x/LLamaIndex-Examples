import os.path
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import FaithfulnessEvaluator

from dotenv import load_dotenv
load_dotenv()

text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=10)

Settings.text_splitter = text_splitter
# Settings.llm = Ollama(model="mistral")
Settings.llm = OpenAI()

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True, transformations=[text_splitter])
    index.storage_context.persist()
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Define the evaluator using the LLM from Settings
evaluator = FaithfulnessEvaluator(llm=Settings.llm)

# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

print(response)
print('\n\n\n')

eval_result = evaluator.evaluate_response(response=response)
# print(eval_result)