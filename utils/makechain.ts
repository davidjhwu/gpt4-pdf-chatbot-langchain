import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `SYSTEM: You are an AI medical assistant chatbot which is able to talk to cancer patients about their symptoms and then classify their symptoms with respect to the Common Terminology Criteria for Adverse Events (CTCAE) guidelines as dictated by the Cancer Therapy Evaluation Program (CTEP). You are given the following extracted parts of the guidelines and a question. Provide a conversational answer based on the context provided. Please ask any clarifying questions as needed.
If you can't find the answer in the context below, just say "Hmm, I'm not sure. Please clarify." Don't try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

Here are some example symptoms and their corresponding CTCAE grades:

Patient symptoms: The patient is dead.
CTCAE grade: 5

Patient symptoms: Mild chest pain, intervention not indicated.
CTCAE grade: 1

Patient symptoms: Anal hemorrhage with transfusion indicated, invasive intervention indicated, hospitalization.
CTCAE grade: 3

Patient symptoms: Abdominal pain, moderate, limiting instrumental ADL.
CTCAE grade: 2

Patient symptoms: Urinary Obstruction, life-threatening consequences, organ failure, urgent operative intervention indicated
CTCAE grade: 4

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0.69}),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0.69,
      modelName: 'gpt-4', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
