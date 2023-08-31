# DeepLearning.ai course with Semantic Kernel

A personal project to translate  the [DeepLearning.ai's Langchain course](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) to use [Microsft's Semantic Kernel framework](https://github.com/microsoft/semantic-kernel/). 

The original Langchain based notebooks are in the [Langchain](langchain) directory. 

The lessons include the following:

|    |         |             |
| -- | ------- | ----------- |
| # | Example | Description |
| 1 | L1-SK-Model_prompt_parser.ipynb | Shows basic of prompt templating and parsing output |
| 2 | L2-SK-Memory.ipynb | Shows augmenting LLM with memory. Volatile memory is used for simplicity |
| 3 | L3-SK-Chains.ipynb | Demonstrate a simple sequential chain and using context memory for more complex chains/graphs |
| 4 | L4-SK-CreateDB.ipynb <br /> <br /> L4-SK-QnA.ipynb |Load a CSV file into a locally persisted Chroma DB with embeddings <br /> <br /> Run RAG based Q&A summary with markdown output  generation by the LLM assisted by retrievals from the Chroma vector store |
| 5 | L5-SK-Evaluation.ipynb | Evaluating outputs from the RAG based Q&A with combination of manual evaluation samples and evaluation question and answer generated by LLM|
| 6 | L6-SK-Agents.ipynb | Create an agent using the SK's planner feature with a few builtin or sample skills |

## Dependencies

* pip install semantic-kernel
* If you are using Chroma as the vector store you need to ```pip install chromadb```. You may need a compatible C++ compilers like the latest gcc for this install to work. Chroma was tested only on WSL (and not Windows native) where you may need to run ```sudo apt-get install build-essential -y```

**Acknowledgements:** The notebooks and code from the Langchain deeplearning.ai course above was used as a starting point for this repo.


