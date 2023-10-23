import os, cohere, openai
import openai, nest_asyncio, warnings
from qdrant_client.http import models
from qdrant_client import QdrantClient
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

from dotenv import load_dotenv
load_dotenv()
# Disable all warnings
warnings.filterwarnings("ignore")
# Enable asyncio
nest_asyncio.apply()


def metadata_creator(llm, embed_model, loaded_doc, separator:str="\n\n", chunk_size:int=1024, chunk_overlap:int=128):
    TITLE_NODE_TEMPLATE = """\
    Context: {context_str}. Give a title that summarizes all of \
    the unique entities, titles or themes found in the context in Vietnamese. Title: """
    TITLE_COMBINE_TEMPLATE = """\
    {context_str}. Based on the above candidate titles and content, \
    what is the comprehensive title for this document? Answer in Vietnamese. Title: """
    QAE_TEMPLATE = f"""\
    {{context_str}}. Given the contextual information, \
    generate 5 questions this document can provide \
    specific answers in Vietnamese to which are unlikely to be found elsewhere: \
    """
    SUMMARY_EXTRACT_TEMPLATE = """\
    Here is the content of the section: {context_str}. \
    Summarize the key topics and entities of the section in Vietnamese. Summary: """
    
    # Create a metadata extractor
    metadata_extractor = MetadataExtractor(
        extractors=[
            TitleExtractor(nodes=3, 
                        llm=llm,
                        node_template=TITLE_NODE_TEMPLATE,
                        combine_template=TITLE_COMBINE_TEMPLATE,
                        ),
            QuestionsAnsweredExtractor(questions=3, 
                                    llm=llm,
                                    prompt_template = QAE_TEMPLATE,
                                    ),
            SummaryExtractor(summaries=["self"], #["prev", "self", "next"]
                            llm=llm,
                            prompt_template=SUMMARY_EXTRACT_TEMPLATE,
                            ),
            KeywordExtractor(keywords=5, llm=llm),
        ],
    )

    # Create a node parser
    text_splitter = TokenTextSplitter(separator=separator, 
                                      backup_separators = ['\n', '. '],
                                      chunk_size=chunk_size, 
                                      chunk_overlap=chunk_overlap)
    node_parser = SimpleNodeParser(
        text_splitter=text_splitter,
        metadata_extractor=metadata_extractor,
    )
    
    # Create a service context
    service_context = ServiceContext.from_defaults(llm=llm, node_parser=node_parser, embed_model=embed_model)
    
    # Create indexer
    index = VectorStoreIndex.from_documents(
        documents=loaded_doc,
        service_context=service_context,
        show_progress=True,
    )
    return index

def qdrant_collection_def(qdrant_url, qdrant_api_key, qdrant_collection, embedding_dim):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    try:
        client.create_collection(
            collection_name=qdrant_collection,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE),
        )
    except:
        pass
    return client

def vectorizer(model:str='azureopenai', embedding_text:str=None):
    if model == 'openai':
        embedding_vector = openai.Embedding.create(
                                input = embedding_text,
                                model="text-embedding-ada-002",
                                )['data'][0]['embedding']
        return embedding_vector
    
    if model == 'azureopenai':
        embedding_vector = openai.Embedding.create(
                    input=embedding_text,
                    engine="vpi-embedding-ada-002"
                )['data'][0]['embedding']
        return embedding_vector
    
    if model == 'cohere':
        co = cohere.Client(os.environ['COHERE_API_KEY'])
        embedding_vector = co.embed(
                            texts=[embedding_text],
                            model='multilingual-22-12',
                            )
        return embedding_vector
    
def qdrant_uploader(client, embedding_model, qdrant_collection, docs):
    points=[]
    for k, doc in docs.items():
        id = k
        embedding_text = doc.metadata['document_title'] + "\n" +\
                        doc.get_content().replace("\n\n", "\n") + "\n" +\
                        doc.metadata['document_title'] + "\n" +\
                        doc.metadata['section_summary'] + "\n" +\
                        doc.metadata['questions_this_excerpt_can_answer'] + "\n" +\
                        doc.metadata['excerpt_keywords']
        point = models.PointStruct(
                id=id,
                vector=vectorizer(model=embedding_model, embedding_text=embedding_text),
                payload={
                    "page_content": doc.get_content(),
                    "file_name": doc.metadata['file_name'],
                    "document_title": doc.metadata['document_title'],
                    "section_summary": doc.metadata['section_summary'],
                    "excerpt_keywords": doc.metadata['excerpt_keywords'],
                    "questions": doc.metadata['questions_this_excerpt_can_answer'],
                },
            )
        points.append(point)
        print(f"Processing Document ID: {id}, File Name: {doc.metadata['file_name']}")
    client.upsert(collection_name=qdrant_collection, points=points)
    print("Qdrant Upserting Finished!")