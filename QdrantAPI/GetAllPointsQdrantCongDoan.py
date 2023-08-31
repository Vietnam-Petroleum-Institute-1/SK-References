from qdrant_client import QdrantClient

def main():
    CD_QDRANT_URL = 'https://2aaeafec-b03e-4545-9b12-f8f806ad320a.eu-central-1-0.aws.cloud.qdrant.io:6333'
    CD_QDRANT_API_KEY = 'QoB7detTXir9bCdMhGNP9tPdWs61VUWChrH9tROY6YXgo4wtD6LNCg'
    collection_name = 'context'
    
    qdrant_client = QdrantClient(url=CD_QDRANT_URL, api_key=CD_QDRANT_API_KEY)
    
    all_points = qdrant_client.scroll(
                    collection_name=collection_name, 
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                )
    
    full_txt = ''
    for i in range(len(all_points[0])):
        full_txt += all_points[0][i].payload['page_content'] + '\n'
    
    response = {}
    response['context'] = full_txt
        
    return response
    