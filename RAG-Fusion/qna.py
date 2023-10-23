import tiktoken, openai
from datetime import datetime


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def responding_openai(query, _search_info):
    prompt = f"""
        You will be provided with the question which is delimited by XML tags and the \
        context delimited by triple backticks. Answer the question in Vietnamese using only provided context.\n
        <tag>{query}</tag>
        ````\n{_search_info}\n```
        """
    prompt_token_length = tiktoken_len(prompt)
    if prompt_token_length > 16000:
        print("Length of Prompt exceeds the limitation of LLM input. Task closed!")
    else:    
        _start = datetime.now()
        _sys_messages = [{"role": "system", "content": "You are a helpful assistant that gives a comprehensive answer  \
                        from the given information"},
                        {"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=_sys_messages,
                max_tokens = 16000 - prompt_token_length, #Maximum length of tokens is 4096 included Prompt Tokens
                n=1,
                temperature=0.1,
                top_p=0.7,
            )
        results = response.choices[0].message.content
        chatgpt_tokens = response.usage.total_tokens
        #Response Time (s)
        chatgpt_response_time = (datetime.now() - _start)
        
    return results, chatgpt_response_time