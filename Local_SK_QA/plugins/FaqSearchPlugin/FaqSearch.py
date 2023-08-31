import os, cohere
from qdrant_client import QdrantClient
from semantic_kernel.skill_definition import (
    sk_function,
    sk_function_context_parameter,
)
from semantic_kernel.orchestration.sk_context import SKContext

# Load environment variables
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

class FAQQdrantSearch:
    """
    Description: Query information from Qdrant Vector Database
    """

    @sk_function(
        description="Ask Qdrant Vector database to receive some information about a specific FAQ.",
        name="faqQuery",
    )
    @sk_function_context_parameter(name="query", description="The raw question for querying")
    @sk_function_context_parameter(name="input", description="The topic of the input question")
    def faq_query(self, context: SKContext) -> str:
        if str(context['input']).strip() == 'Congdoan':
            qdrant_client = QdrantClient(url=os.environ['CD_QDRANT_URL'], api_key=os.environ['CD_QDRANT_API_KEY'])
        elif str(context['input']).strip() == 'Quyhoach':
            qdrant_client = QdrantClient(url=os.environ['QH_QDRANT_URL'], api_key=os.environ['QH_QDRANT_API_KEY'])
        else:
            print('Không xác định được chủ đề câu hỏi. Hãy đặt câu hỏi rõ ràng hơn!')
        cohere_client = cohere.Client(api_key=os.environ['COHERE_API_KEY'])

        results = qdrant_client.search(collection_name='faq',
                    query_vector=cohere_client.embed(texts=[context['query']],
                                                    model='multilingual-22-12',
                                                    ).embeddings[0],
                    limit=1
                    )
        #Return
        matched_ans = results[0].payload['metadata']['answer']
        matching_score = results[0].score

        #Update context
        context['matched_ans'] = matched_ans
        context['matching_score'] = matching_score
        
        #Print
        print(matching_score)
        print(matched_ans)
        
        return matched_ans