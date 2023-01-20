# Leveraging ChatGPT for <br> Business and Organizational Purposes    

Since its introduction in November 2022, the ChatGPT chatbot has captivated the world by answering questions on virtually any subject in human-like fashion.  Not to mention its ability to compose poems. Or write computer code based on natural-language instructions.  

ChatGPT will no doubt have a huge impact on the public -- as well as on businesses and other institutions.

For organizations, the trick will be to leverage ChatGPT's extraordinary powers across specific domain areas, such as industries or corporate functions.

An insurance company, for example, might customize  ChatGPT's answers to information in its policy documents. The answers could be used internally to train and inform service reps, as well as with customers directly via a website chatbot. The frictional costs -- and frustration levels -- of finding the right information would plummet.

Industries and functions that are knowledge-intensive (e.g., healthcare, professional services, education) and service-intensive (IT, legal, marketing and sales) will likely benefit most from ChatGPT’s powers .     

But almost all companies and functions have significant knowledge management needs, and ChatGPT could help them improve how they operate and serve customers -- perhaps dramatically.

## Building Domain-Specific Capabilities
Developed by OpenAI, a research company partially owned by Microsoft, ChatGPT was "trained" on a massive trove of text data on the internet up through 2021 -- some 300 billion words from web pages, books and other documents. 

Due to the training cut off, ChatGPT knows nothing about events that occurred in 2022 and later. Nor does it know anything about company and organization documents that were not available to it in its training. 

But through an API (Application Programming Interface, which lets different computer programs talk with each other), ChatGPT can process and incorporate new information in real-time. This feature enables it to stay up-to-date with the latest developments or specific knowledge in an industry or field.

To demonstrate how this might work, I built a domain-specific [chatbot](https://huggingface.co/spaces/robjm16/domain_specific_ChatGPT). In my example, I took the 2023 investment outlook summaries posted to the web  by Morgan Stanley [(here)](https://www.morganstanley.com/ideas/global-investment-strategy-outlook-2023), JPMorgan [(here)](https://www.jpmorgan.com/insights/research/market-outlook) and Goldman Sachs [(here)](https://www.goldmansachs.com/insights/pages/gs-research/macro-outlook-2023-this-cycle-is-different/report.pdf) and combined them into one 4,000 word document. 

Through a process described in more detail below, the investment outlook information was fed into ChatGPT and became the basis for responses to questions such as: "What does Goldman see happening with inflation in 2023?" and "What is the outlook for the bond market?" (Note: Most of the my code was adapted from OpenAI's [cookbook](https://github.com/openai/openai-cookbook) of code examples for working with ChatGPT.)

Below is an overview of what I discovered through the development process, written for both technical and general audiences. 

(By the way, GPT stands for “Generative Pretrained Transformer,” a technical reference to the AI model.)

## ChatGPT’s Many Uses 
ChatGPT’s capabilities go well beyond what traditional chatboxes offer: 
- It can be used to draft copy for marketing materials, blog posts and product descriptions. 
- It can edit, summarize or translate any text, and write in almost any voice (e.g., as a pirate).  
- It can be used for text classification – for example, whether tweets about my organization were positive or negative last week.      
- It can search documents via “semantic search,” which captures the broader intent of your search query, not just the exact words. 

On the computer coding side: 
- It can convert written instructions into computer code.
- It can auto-complete code.
- It can explain and document your code. 
- It can write test cases and fix bugs.
- It can convert between coding languages (e.g., Java to Python). 
   

## Two Key Mechanisms: Prompt and Completion 
When interacting with ChatGPT, either through a simple web interface or through computer code via the API, the prompt and completion mechanisms are key.

The prompt is an input mechanism where you place your question or request, as well as any context, including domain-specific content (e.g., the 2023 investment outlooks) and other instruction (e.g., respond in a certain format). 

The completion mechanism is ChatGPT’s response to your prompt.  It answers your question or request.  Importantly, it contains a parameter called “temperature,” which controls how creative ChatGPT should be in responding to your prompt.  A lower temperature means ChatGPT should be conservative, sticking to the most factual information -- that is, not try to respond when it is not sure. 

## Domain-Specific Uses: Technical Approaches
There are essentially three ways to interact with ChatGPT for domain-specific purposes:
1. Use as is:  The first approach is to use ChatGPT as is.  For example, ChatGPT has well-honed classification capabilities, and it may not benefit much from domain-specific examples.  If you want to use ChatGPT to rate sentiments in hotel or restaurant reviews from the internet, its inherent capabilities should work fine. 

2. Inject content into prompts: The second approach, which I took in my demo example, is to inject domain-specific context into your prompt.  In this  scenario, ChatGPT uses its well-practiced natural language capabilities, but then looks to your specific content when formulating an answer.

      
3. Fine-tune a model:  Currently, only the previous and less powerful version of ChatGPT’s neural network model (GPT2) is available to install and use in your own environment.  With GPT2 and some other pre-trained libraries, you can alter aspects of the model in a process called transfer learning, and train it on your domain-specific content.  

    The newest model (GPT3) can be accessed via the OpenAI API.  You can “fine tune” it on your content and save a new  version of it (at OpenAI) for future use via the API.  But you cannot fundamentally alter and retrain it in the traditional machine learning sense.  One reason why is the sheer size of the pre-trained model -- the time and cost of retraining would be prohibitive for virtually all users.

    Instead, with GPT3, you create a new version of the model and feed it your domain-specific content.  The model then runs in the background, seeking to maximize correct answers by updating some of the model’s parameters (see discussion of neural networks below).  When complete, it creates a proprietary version of the model for your organization. 

The second and third approaches above use a technique known as “in-context” learning.  The key difference between the two is that the second injects the domain- specific content in real time into the prompt whereas approach three tailors the model to your needs and produces a reusable customized model, with potentially more accurate results.  With approach two, the base model is used unchanged and the model retains no "memory" of the injected content, outside of the current session.  

## Word Embeddings: 4,000 Shades of Meaning 
When ChatGPT receives a question, it maps each word or word fragment to a token, a unique numerical identifier.  With ChatGPT, each token represents approximately 0.75 words.  (The math is important due to usage limits on ChatGPT.)

Each token also gets a numerical representation of the word or word fragment called an "embedding." For example, the word "queen" can be represented by a series of numerical sequences capturing how close the word is semantically to words such as "king," "female” and “leader."  The embedding also captures syntax and context.  The actual embedding for a token in the text might look like this: [0.016102489084005356, -0.011134202592074871, …, 0.01891878806054592].  

In ChatGPT's case, each token has 4,096 data points or dimensions associated with it. In addition, ChatGPT's artificial intelligence model -- a deep neural network -- pays attention to words that come before and after, so it holds on to context as it "reads in" new words. 

## GPT3: One of World’s Largest Neural Networks 
Neural networks are often described as brain-like, with “neurons” and their connecting “synapses.”  In the simple example below, the far left layer takes in input (the word-derived tokens) and the far right layer is the output (the answer or response).  In between, the input goes through many layers and nodes, include the embedding, depending on the complexity of the model.  This part is “hidden” in that what each node represents is not easily discernable.  

The lines between the model's nodes (similar to synapses connecting neurons in the brain), receive a mathematical weighting that maximizes the chances that the output is correct.  These weightings are called parameters.   

![image](https://github.com/robjm16/domain_specific_ChatGPT/blob/main/basic_nn.png?raw=true)  

  
The ChatGPT model (GPT3) has 175 billion potential line weightings or parameters, but not all of them “fire” depending on the prompt.  By contrast, GPT2 has 1.5  billion parameters.  

The ChatGPT model also has an “attention” mechanism that allows it to differentially weight the importance of different parts of the input text, leading to a more coherent and fluent response.  

In addition, the ChatGPT model was partially trained on how actual human beings rated answers, helping to make responses not just correct but more human friendly.  

## ChatGPT in Action: My Investment Outlook Example 
The first step in leveraging ChatGPT on domain-specific content is to gather the content and pre-process as needed.

The ChatGPT API has limits on the amount of work it will do for free. Accordingly, I limited my example to about 4,000 words containing the three banks' investment outlooks. I further arranged the content into about 30 paragraphs.

There is a limit of 2,048 tokens – or about 1,500 words – for both the prompt and completion.  While my document is 4,000 words, only the most relevant sections are fed into the prompt, thus staying within the token limit.  

The document’s 30 paragraphs are first sent out to the ChatGPT API to get word embeddings. When a question is asked, that question also gets its respective embeddings via the API.

Next, computer code in my environment (not via the API) compares the question to the content in the 30 paragraphs. It then picks the best-fit paragraphs based on how close the question is semantically to each paragraph (by doing a lot of math around their respective embeddings).

The best paragraphs are then attached to the question as "context" and fed back to ChatGPT via the API for an answer. My program also instructs ChatGPT to say, "Sorry, I don't know," if it is asked a question where it does not have good information, thus reining in ChatGPT's tendency to answer even when unsure. 

Lastly, ChatGPT combines the question, the added domain content and the model's inherent natural language processing skills to produce a response.

Below is an example of a question and response within the interface:

![image](https://github.com/robjm16/domain_specific_ChatGPT/blob/main/interface_example.png?raw=true)
 
## The ChatGPT Ecosystem 
OpenAI was founded in 2015 by a group that includes Elon Musk.  As mentioned earlier, Microsoft is an investor and key partner.  

Microsoft plans to incorporate ChatGPT into many of its offerings.  For example, it could be integrated with  Microsoft Word and PowerPoint, for help in writing, editing and summarization. It could be used to augment Microsoft’s Bing search engine, providing direct answers to questions along with a more semantic search engine. ChatGPT’s coding assistance abilities could be integrated with Microsoft’s Visual Studio code editing product. (Microsoft offers Github Copilot, a code auto-completion tool, and some coders are already using Copilot and GPT3 in tandem to improve their productivity.) Lastly,  Microsoft Azure’s cloud computing services are already incorporating GPT3 -- for example, helping large companies fine-tune ChatGPT on domain-specific content. 

The other large cloud providers – Google and Amazon Web Services (AWS) – will no doubt integrate GPT3 into their AI offerings, while continuing to enhance their own AI models.

Google’s CEO has reportedly called a “code red” following the release of ChatGPT, challenging the company to quickly incorporate Google’s own ChatGPT-like models into its dominant search platform. 

Google, in fact, developed several of the most powerful “large language models” similar to GPT3 (they go by the names BERT, T5 and XLNet). 

Meta is also a key player, with Facebook’s RoBERTa model.  

AWS’s suite of AI services is called SageMaker.  It includes pre-built algorithms and enables companies to quickly build, train and deploy machine learning models.

Another player is Hugging Face, which hosts a popular community website for sharing open-source models and for quickly prototyping and deploying natural language processing models. The platform includes a library of pre-trained models, a framework for training and fine-tuning models, and an API for deploying models to production.  You can access and use GPT2 through Hugging Face (with GPT3 available through the OpenAI API.) 

## Data Security
Each organization will have to make its own security judgments around using ChatGPT, including server location, VPN, firewall,  encryption and data security issues. 

Based on a series of questions to the chatbot itself,  ChatGPT says that information shared as context in the public chatbot (and information served back in response) could potentially make its way into subsequent training of the base GPT model.    

ChatGPT notes, however, that it has guidelines to guard against confidential or personal information being used in training.  Further, it says that, given the massive size of GPT's training data, it is unlikely that a small amount of organizational data could affect the base model's training.  

The fine-tuned model is different.  OpenAI does not have access to the prompts you use to fine-tune your version of the model and thus could not use them to train the base model. 

In addition, your organization can purchase GPT3 licenses for on-premises deployment or a "fully managed" enterprise solution hosted on the Microsoft Azure cloud.

Bottom line: organizations will need to think carefully about restricting or sanitizing some inputs and choosing the right fine-tuning and security arrangements.



