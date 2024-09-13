from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages

class ChatBot:
    def __init__(self, model_name="gpt-3.5-turbo-0125", temperature=0, session_id='harsh'):
        """
        Initializes the chatbot with a given model, temperature, and session ID.
        """
        self.chat = ChatOpenAI(model=model_name, temperature=temperature)
        self.output_parser = StrOutputParser()
        self.prompt = ChatPromptTemplate(
            [
                ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        )
        self.trimmer = trim_messages(
            max_tokens=16385,
            strategy="last",
            token_counter=self.chat,
            include_system=True,
        )
        self.chain = self.trimmer | self.prompt | self.chat | self.output_parser
        self.store = {}
        self.session_id = session_id
        self.history_chain = self.get_history_conversation_chain()

    def get_session_history(self, session_id):
        """
        Fetch or create an in-memory chat history for the session.
        """
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def get_history_conversation_chain(self):
        """
        Create a RunnableWithMessageHistory chain that handles chat history.
        """
        return RunnableWithMessageHistory(
            self.chain,
            get_session_history=self.get_session_history
        )

    def get_response(self, query):
        """
        Process the user query, add to chat history, invoke the chain, and return the response.
        """
        if query.strip():
            human_message = HumanMessage(query)
            return self.history_chain.invoke(
                [human_message],
                config={"configurable": {"session_id": self.session_id}}
            )
        return "Invalid input. Please enter a valid query."

    def chat_loop(self):
        """
        Main loop for the chatbot. Keeps taking user queries until 'q' is entered to quit.
        """
        print(f"You've logged in as {self.session_id}")
        while True:
            query = input("Your query (Enter 'q' to quit): ").strip()

            if query.lower() == 'q':
                print("Ending the chat")
                break

            response = self.get_response(query)
            print(response)


if __name__ == "__main__":
    load_dotenv()

    bot = ChatBot()
    bot.chat_loop()
