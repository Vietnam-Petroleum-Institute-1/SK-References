import os
import cohere
from underthesea import word_tokenize
from qdrant_client import QdrantClient
from anthropic import Anthropic
from langchain.vectorstores import Qdrant
from langchain.embeddings import CohereEmbeddings

#FACTS MATCHING-------------------------------------------------------------------
def facts_matching(query, cohere_api_key, qdrant_client):
    cohere_client = cohere.Client(api_key=cohere_api_key)
    
    results = qdrant_client.search(collection_name='faqVieProcessed',
                query_vector=cohere_client.embed(texts=[query],
                                                model='multilingual-22-12',
                                                ).embeddings[0],
                limit=1
                )
    #Return
    matched_ans = results[0].payload['metadata']['answer']
    matching_score = results[0].score
    return matched_ans, matching_score

#DATABASE-SEARCHING----------------------------------------------------------
def searching(query, vdatabase):
    _query = word_tokenize(query, format="text")
    search_results = vdatabase.similarity_search_with_score(_query, k=3)
    return search_results

def prompting(search_results, query, matched_ans, user_prompt:str=None):
    _search_info = " --- " + " --- ".join([search_results[i][0].page_content 
                                    for i in range(len(search_results))]) + " --- "
    raw_prompt = "You will be provided with the question which is delimited by XML tags and the context delimited by triple backticks. The context contains some long paragraphs and 1 reference which delimited by triple dash."
    prompt = raw_prompt + f"""
    <tag>{query}</tag>\n
    ````\n{_search_info}```\n{matched_ans}\n```
    """
    if user_prompt is not None:
        prompt = prompt + '\n\n Hãy tuân theo các yêu cầu sau đây:\n' + user_prompt
    
    return prompt.replace("_"," ")

#---CLAUDE-RESPONSE-----------------------------------------------
def responding_claude(prompt, ANTHROPIC_API_KEY):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    HUMAN_PROMPT = f"\n\nHuman: {prompt}"
    AI_PROMPT = "\n\nAssistant:"
    completion = client.completions.create(
        model="claude-2",
        max_tokens_to_sample=2000,
        temperature=0.1,
        prompt=f"{HUMAN_PROMPT} {AI_PROMPT}",
    )
    results = completion.completion
    return results.replace('<tag>','').replace('</tag>','')
def main(query, qdrant_url, qdrant_api_key, user_prompt:str=None):
    ANTHROPIC_API_KEY = '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'
    cohere_api_key= '4ECOTqDXJpIYhxMQhUZxY12PPSqvgtYFclJm4Gnz'
    collection_name='contextVieProcessed'
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    vdatabase = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embeddings)
    #------------------------------------------------------------------------
    matched_ans, matching_score = facts_matching(query, cohere_api_key, qdrant_client)
    if matching_score > 0.96:
        print(f'Câu hỏi: {query}\n\nTrả lời: {matched_ans}\n\nĐộ chính xác: {round(matching_score,2)}')
    else:
        search_results = searching(query, vdatabase)
        if user_prompt is not None:
            prompt = prompting(search_results, query, matched_ans, user_prompt)
        else:
            prompt = prompting(search_results, query, matched_ans)
        results = responding_claude(prompt, ANTHROPIC_API_KEY)
        print(f'Câu hỏi: {query}\n\nTrả lời: {results}')
        print('\n\n===NGUỒN THAM KHẢO==============================================')
        for i in range(len(search_results)):
            reference = search_results[i][0].page_content.replace('_',' ')
            print(f'Tham khảo {i+1}:\n{reference}')
        print('\n==================================================================')
        print(f'Tham khảo FAQ:\n{matched_ans}')
