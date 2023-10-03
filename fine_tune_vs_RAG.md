# Domain-Specific AI: Untangling the Options

For many companies, the value of ChatGPT and similar AI models depends on how well they "learn" domain-specific information.  Accounting firms, for example, need AI systems to answer complex tax-related questions with high precision.  But while base AI models provide impressive general language capabilities, they need some level of customization to deliver the domain-specific results most companies require.  

There are several ways you can work with ChatGPT and similar large language models (LLMs) to achieve domain- or task-specific goals.  

The simplest approaches are:  
- **Prompting**:  Thoughtfully formulating input phrases to get desired responses from LLMs.
- **One- or few-shot learning**:  Inclusion of one or a small number of examples within the prompt, showing both input and desired output.  The goal is to adapt the model to new tasks using minimal additional data (just examples).

These are basic starting points. They can sharpen results, but only to a degree.  The reason is simple:  general-purpose LLMs have not been trained deeply in most domain areas -- and certainly not on your company's proprietary knowledge base.  

At the other end of the customization continuum is:  

- **Pre-training**: Foundational stage where an LLM is trained from scratch on a vast body of text, to grasp semantics, syntax and knowledge. Building an LLM from scratch requires significant computational resources, extensive data, skilled experts, months of work and an investment of millions to tens of millions of dollars. (As the performance of smaller LLMs improves, more companies might opt for them, potentially reducing pre-training costs.)

ChatGPT is a general purpose LLM, but an LLM can also be pre-trained on domain-specific information.  An example is BloombergGPT, which was trained extensively on financial information. But for most companies today, domain-specific customization comes down to two approaches that stop short of full pre-training:  

- **Retrieval Augmented Generation (RAG)**:  Adds a knowledge retrieval step to the prompt engineering process. The model fetches relevant chunks of information from your company's knowledge base and appends them as context to a prompt.  Implementing RAG requires a well-structured knowledge base and a mechanism to effectively retrieve relevant content.  The approach has moderate to high complexity and can take days to months to implement.  

- **Fine-tuning**:  Process of adapting a pre-trained LLM (called a foundational model) by further training it on a domain-specific dataset. It requires a robust body of text and hundreds to many thousands of training examples (showing input and desired output), as well as specialized knowledge of how to tune LLMs for optimal performance.  Like RAG, fine-tuning comes with moderate to high complexity but generally requires less time and investment (depending on facts and circumstances).

## RAG and Fine-Tuning: A Deeper Dive
Below are key considerations when implementing RAG and fine-tuning. An important note: the two approaches need not be mutually exclusive. RAG can be helpful in ring-fencing information that should be the basis of a response, while fine-tuning can make the model generally more conversant in a domain area, and the responses better tailored to your needs.  

- **Hallucinations**:  This is the well-known tendency of LLMs to "make up" an answer even when unsure of the facts.   RAG can address this by directing the model to only respond based on the provided context, thus largely eliminating hallucinations.  With fine-tuning, the additional training can cut down on hallucinations, but it will not provide the certainty of RAG.
- **Answer accuracy**: One drawback of RAG is that it might provide too narrow an answer. That is because the retrieved text may be limited in size (models have maximum prompt input lengths) or because the retrieval mechanism failed to deliver the most relevant chunks of context. The best answer, for example, might be derived across a wide swath of texts - too many chunks of text for a prompt. On the other hand, RAG offers the ability to attach metadata (e.g., region, industry), which can be helpful in properly focusing responses. RAG also tends to deliver more detailed responses.
- **Tone and behavior**: With fine-tuning, you can train a model to reply with a more uniform tone and structure, tailored to your needs.
- **Source citation**:  In some domain areas (e.g., the law), the ability of a model to provide specific citations of facts informing an answer is very important.  RAG offers a straightforward path to providing citations.
- **Training risk**:  With fine-tuning, there is a chance of going too far and diminishing  a capability of the foundational model.   You are "opening up the model's hood" – at least partially – which carries risks.
- **Stability and evolution**:  RAG is considered a relatively stable approach, as it leverages mature underlying technologies (databases and retrieval mechanisms).  LLM technology, by contrast, is rapidly evolving rapidly.  Repeated fine-tuning may be needed as new and improved versions become available.
- **Costs and complexity**:  With RAG, there is added effort in carefully maintaining a domain-specific database (typically via a vector database such as Pinecone).  It can also cost more in the end as the prompts are necessarily longer and more costly computationally.  Generally, fine- tuning requires simpler architecture and less maintenance effort.

Regardless of which methods you choose, you will ultimately need to decide on a foundational model and when in the cycle to adopt.  You will also need to think through how to maintain large, domain-specific bodies of content, training data, model performance histories and prompt templates.  Thankfully, developer tools are emerging (e.g., LangChain) that simplify the process of RAG and fine-tuning.  

## Broader considerations
Decisions around domain-specific AI cannot take place in a vacuum.  Among the broader considerations:  
- **Overall AI strategy**:  Where are the biggest opportunities (e.g., what business areas, functions, tasks)?   Costs vs benefits?  How to start (e.g., pilots, prototypes)
- **Security/privacy/compliance/ethics**:  How to ensure AI efforts are consistent with your standards?  How to prevent data leakage? Inappropriate responses?
- **Vendor strategy/ecosystem/lock**:  Who are the likely winners?  How much to consolidate, where and with whom?  Firewall issues?   In-house vs. cloud?
- **Integration**: How do potential AI technologies integrate with our other technologies?
- **Scalability**:  How well will any solutions scale to our company – now and in future?
- **Technology maturity**:  When is the right time to lock in different technologies, given rapid evolution?
- **Skill availability**: Do we have the right technical skills?
- **Change management**:  What will be the impact on people, jobs and processes?  How do we get there? Leaders, stakeholders, etc.?

Getting the right domain-specific AI strategy is essential – and tricky.  The opportunities are huge, but so are the unknowns.  In any case, the right strategy must be a blend of deep AI-specific thinking and a broader view of your organization's IT and business strategies.
