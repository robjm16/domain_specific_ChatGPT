# Leveraging ChatGPT for <br> Business and Organizational Purposes    

Since its launch in November 2022, ChatGPT has captivated the world by answering questions on virtually any subject in human-like fashion.  Not to mention its ability  to compose poems in seconds. Or write computer code based on plain language instructions.  

ChatGPT will no doubt have a huge impact on the public -- as well as on companies and other institutions.

For organizations, the key will be to leverage ChatGPT's extraordinary powers across specific domain areas, such as industries or corporate functions.

An insurance company, for example, might customize ChatGPT to answer questions on all its policy documents. The modified version could be used internally to train and inform service reps and, eventually, with customers directly via its app or website. The frictional costs of getting the right information in the right format at the right time would trend to zero. 

Industries and functions that are knowledge-intensive (e.g., healthcare, professional services, education) and service-intensive (IT, legal, marketing and sales) will likely benefit most from ChatGPT’s powers.     

But almost all organizations and functions have significant knowledge management needs, and ChatGPT could help improve how they operate and serve customers -- perhaps dramatically.


## Building Domain-Specific Capabilities
Developed by OpenAI, an artificial intelligence  research company, ChatGPT was "trained" on a massive trove of text data on the internet up through 2021 -- some 300 billion words from web pages, books and other documents. (The "GPT" in ChatGPT stands for “Generative Pretrained Transformer,” a technical reference to the AI model.)

Due to the training cut off, ChatGPT knows little or nothing about events that occurred in 2022 and later. Nor does it know anything about organizational documents that were not available to it in its training. 

But through an API (Application Programming Interface, which lets computer programs talk with each other), organizations can incorporate new information into ChatGPT. This feature enables it to stay up-to-date with the latest developments or specific knowledge in an industry or field.

To demonstrate how this might work, I built a simple domain-specific [chatbot](https://huggingface.co/spaces/robjm16/domain_specific_ChatGPT). In my example, I took the 2023 investment outlook summaries posted to the web by Morgan Stanley [(here)](https://www.morganstanley.com/ideas/global-investment-strategy-outlook-2023), JPMorgan [(here)](https://www.jpmorgan.com/insights/research/market-outlook) and Goldman Sachs [(here)](https://www.goldmansachs.com/insights/pages/gs-research/macro-outlook-2023-this-cycle-is-different/report.pdf) and combined them into one 4,000 word document. 

Through a process described more fully below, the investment outlook information was fed into ChatGPT and became the basis for responses to questions such as: "What does Goldman see happening with inflation in 2023?" and "What is the outlook for the bond market?" 

Most of the [my code]([https://github.com/robjm16/domain_specific_ChatGPT](https://github.com/robjm16/domain_specific_ChatGPT/blob/main/domain_specific_chatbot_prompt_engineering_and_fine_tuning_GITHUBFINAL.ipynb)) was adapted from OpenAI's [cookbook](https://github.com/openai/openai-cookbook) of code examples for working with ChatGPT.  

Below is an overview of what I discovered through the development process and related research. 

## ChatGPT’s Many Uses 
ChatGPT’s capabilities go well beyond what traditional chatboxes offer: 
- It can draft copy for marketing materials, blog posts and product descriptions. 
- It can edit, summarize or translate any text, and write in almost any voice (e.g., ad copy tone).  
- It can be used for text classification – for example, whether tweets about your organization were positive or negative last week.      
- It can quickly organize unstructured information, such as a doctor's diagnostic notes.  

On the computer coding side: 
- It can convert written instructions into computer code.
- It can explain and document your code. 
- It can convert between coding languages (e.g., Java to Python). 
- It can write test cases and help fix bugs.

## Two Key Mechanisms: Prompt and Completion 
When interacting with ChatGPT, either through a web interface or through computer code via the API, the prompt and completion mechanisms are key.

The prompt is an input mechanism into which you place your question or request, as well as any context, including domain-specific content and other instructions (e.g., respond in a certain format). 

The completion mechanism is ChatGPT’s response to your prompt.  It answers your question or request.  Importantly, it contains a parameter called “temperature,” which controls how creative ChatGPT should be in responding to your prompt.  A lower temperature means ChatGPT will be conservative, sticking to the most factual information and not trying to guess if unsure. 

## Domain-Specific Uses: Technical Approaches
There are three ways to interact with ChatGPT for domain-specific purposes:
1. Use as is:  The first approach is to use ChatGPT as is.  For example, ChatGPT has well-honed classification capabilities, and it may not benefit much from domain-specific examples.  If you want to use ChatGPT to classify or summarize online review sentiment about your business, its inherent capabilities should work fine. 

2. Inject content into prompts: The second approach is to inject domain-specific context into your prompt.  In this  scenario, ChatGPT still fully uses its natural language capabilities, but it looks to your specific content when formulating an answer.
     
3. Fine-tune a model:  Currently, only the previous and less powerful version of ChatGPT’s neural network model (GPT-2) is available to download and use in your own environment.  With GPT-2 and other relatively small pre-trained libraries, you can adapt the model in a process called transfer learning and train the model on your domain-specific content.  

    The newest model (GPT-3) can only be accessed via the OpenAI API.  You can “fine tune” it on your content and save a new version of it for future use.  But you cannot fundamentally modify the model and retrain it in the traditional machine learning sense. 
    
    One reason why is the sheer size of the pre-trained model. The time (weeks or months) and computing costs of fully retraining it would be prohibitive to all but the largest organizations. Further, any significant retraining would run the risk of inadvertently losing some of ChatGPT's powerful capabilities.  

    Instead, with GPT-3, you start with the base model and feed it your domain-specific content in the form of questions and answers. Making matters easier, the model itself can create the questions and answers based off of your content. The model then runs in the background, seeking to maximize its ability to answer correctly by updating some of the model’s parameters (see discussion of neural networks below).  When complete, it creates a proprietary version of the model for future use.  

The second and third approaches are not mutually exclusive. The key difference is that the third approach tailors the model to your information and produces a reusable customized model (more on this later). With approach two, the base model is used unchanged and the model retains no "memory" of the injected content, outside of the current session.  

## Contextual Embeddings: 4,000 Shades of Meaning 
When ChatGPT receives a question and other content as input, it first maps each word or word fragment to  a unique numerical identifier called a token.  With ChatGPT, each token represents approximately 0.75 words.  (The math is important due to usage limits on ChatGPT.)

Each token in the input receives a numerical representation of the word or word fragment called an "embedding." For example, the word "queen" can be represented by a series of numerical sequences capturing how close the word is semantically to words such as "king," "female” and “leader."  The embedding also captures syntax and context.  

ChatGPT then combines all the tokens and embeddings in the input text (let's assume it's a paragraph) and generates a fixed-length numerical representation of the paragraph.  In ChatGPT's case, each input paragraph has 4,096 data points or dimensions associated with it. This is known as "contextual embedding." The actual embedding might look like this: [0.016102489084005356, -0.011134202592074871, …, 0.01891878806054592].  

## GPT-3: One of World’s Largest Neural Networks 
Neural networks are often described as brain-like, with “neurons” and connecting “synapses.”  In the simple example below, the far left layer takes in input (e.g., a paragraph of text) and the far right layer is the output (the answer or response).  In between, the input goes through many layers and nodes, depending on the complexity of the model.  This part is “hidden” in that what each node represents is not easily discernable.  

The lines between the model's nodes (similar to synapses connecting neurons in the brain), receive a mathematical weighting that maximizes the chances that the output will be correct (and errors minimized).  These weightings are called parameters.   

![image](https://github.com/robjm16/domain_specific_ChatGPT/blob/main/basic_nn.png?raw=true)  

  
The ChatGPT model (GPT-3) has 175 billion potential line weightings or parameters, but not all of them “fire” depending on the prompt.  By contrast, GPT-2 has "just" 1.5  billion parameters.  

In addition, the ChatGPT model has an “attention” mechanism that allows it to differentially weight the importance of parts of the input text, leading to a more coherent and fluent response.  

The ChatGPT model was also partially trained on how actual people rated its answers, helping to make responses not just factually correct but more human like.  

## ChatGPT in Action: My Investment Outlook Example 
The first step in leveraging ChatGPT on domain-specific content is to gather the content and pre-process it as needed (e.g., chunking it into sentences or paragraphs).

The ChatGPT API has limits on the amount of work it will do for free. Accordingly, I limited my example to about 4,000 words containing the investment outlooks from the three banks. I further arranged the content into about 30 paragraphs.

There is a limit of 2,048 tokens – or about 1,500 words – for both the prompt and completion.  While my document is 4,000 words, only the most relevant sections are fed into the prompt, thus keeping under the token limit.  

The document’s 30 paragraphs are first sent out to the ChatGPT API to get contextual embeddings. When a question is asked, that question also gets its respective embeddings via the API.

Next, computer code in my environment (not via the API) compares the question to the content in the 30 paragraphs. It then picks the best-fit paragraphs based on how close the question is semantically to each paragraph (by doing a lot of math around their respective contextual embeddings).

The best-fit paragraphs are then attached to the question as "context" and fed back to ChatGPT via the API for an answer. My program also instructs ChatGPT to say, "Sorry, I don't know," if it is asked a question where it does not have good information, thus reining in ChatGPT's tendency to answer even when unsure. 

Lastly, ChatGPT combines the question, the added domain content and the model's inherent natural language processing skills to produce a response.

Below is an example of a question and response within the interface:

![image](https://github.com/robjm16/domain_specific_ChatGPT/blob/main/interface_example.png?raw=true)

After fine tuning and creating a proprietary version of the base model, I found that my new version of ChatGPT could answer questions based off of the newly ingested domain-specific content, even without added context in the prompt.  However, the answers were much less specific than when context was attached, as in approach two.

Next, I combined the two approaches -- fine tuning a custom model while adding best-fit context in the prompt.  

This seemed to work at least as well as approach one (context added but no fine tuning).  But further experimentation and testing would be needed to determine if, in my case, fine tuning added extra value.  

It is important to note that fine tuning does not create a new store of information in the model.  In fine tuning, you typically feed the model hundreds of examples, whereas the base model has been trained on hundreds of millions of documents.  At best, it appears that fine tuning can adjust the model to some degree to your domain area's terminology and specific task instructions. Fine tuning can also work well on very specific classification exercises.  However, to align responses closely with domain content in a question-and-answer model, you should continue to inject the domain content into the prompt.      
    
## The ChatGPT Ecosystem 
OpenAI was founded in 2015 by a group that includes Elon Musk, with Microsoft as an investor and key partner.  

Microsoft plans to incorporate ChatGPT into its product portfolio. For example, ChatGPT could be used in Microsoft Word and PowerPoint, to automate writing, editing and summarization. It could also be used to augment Microsoft’s Bing search engine, providing human-like answers to questions as well as a more semantic-based search. 

ChatGPT’s coding assistance capabilities could be integrated with Microsoft’s Visual Studio code editor. In fact, some coders are already using GPT-3 in tandem with Microsoft's Github Copilot, a code auto-completion tool, and reporting significant gains in personal productivity.  And Microsoft has moved quickly to integrate GPT-3 into its Azure cloud computing services. 

Other large cloud providers – notably Google Cloud Platform and Amazon Web Services (AWS) – also offer fast evolving AI tools. 

Google, in fact, developed several of the most powerful “large language models” similar to GPT-3 (they go by the names BERT, T5 and XLNet). Google’s CEO called a “code red” following the release of ChatGPT, challenging Google engineers to quickly incorporate its ChatGPT-like models into its dominant search platform. 

AWS’s suite of AI services is called SageMaker.  Like the other cloud-based AI toolkits, SageMaker includes pre-built algorithms that enable companies to quickly build, train and deploy machine learning models.

Meta/Facebook and Salesforce have also developed large language models (RoBERTa and CTRL, respectively). 

Another player is Hugging Face, which hosts a popular community website for sharing open-source models and quickly prototyping and deploying  models. You can download and use GPT-2 through Hugging Face (or access GPT-3 through the OpenAI API.) 

## Data Security
Each organization needs to make its own security judgments around using ChatGPT, including server, VPN, firewall,  encryption and data security issues. 

OpenAI says that information shared as input in the public chatbot (as well as the responses) could be used to improve the future performance of the GPT model. It notes, however, that it has guidelines to guard against confidential or personal information being used in training. Further, given the massive size of GPT's training data, OpenAI believes it is unlikely that a small amount of organizational data could affect the base model's training. That said, caution is warranted. 

The fine-tuned model is different.  OpenAI says that it does not have access to the prompts you might use in fine tuning the model, and it could not use them to train the publicly available base model. 

For added control and security, organizations can purchase GPT-3 licenses for on-premises deployment or a "fully managed" enterprise solution hosted on the Microsoft Azure cloud.

## Bottom line
ChatGPT is a disruptive technology, a potential game changer. 

Companies should start by considering its potential strategic impact and by identifying possible use cases.  How might these models disrupt your industry? Where can they be leveraged to dramatically improve knowledge management, customer service or functional processes?  

Organizations can experiment with ChatGPT by developing low-risk prototypes in sandboxed environments. The process will give rise to many questions about restricting or sanitizing inputs, curating domain-specific content, dealing with limitations of the models (e.g., serving up incorrect information), fine tuning, hosting and security. 

It's important to note that ChatGPT-like models are far from being black boxes that operate as autonomous machines. In fact, as a recent McKinsey [report](https://www.mckinsey.com/capabilities/quantumblack/our-insights/generative-ai-is-here-how-tools-like-chatgpt-could-change-your-business) notes, "In many cases, they are most powerful in combination with humans, augmenting their capabilities and enabling them to get work done faster and better."  

Some liken their use to having an army of interns at your side. The interns are super fast, incredibly productive and very smart -- but also overconfident and inexperienced. They need supervision, someone who can catch errors. With that caveat, they could make your marketers, lawyers, software engineers, service reps and other subject matter experts -- and your organization overall -- far more effective and efficient.  
