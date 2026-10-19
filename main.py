import os
from rag_engine import RAGAssistant

def main():
   
    pdf_file = os.path.join("data", "sample.pdf")

    if not os.path.exists(pdf_file):
        print(f" Error: '{pdf_file}' file not found!")
        print("Please place your PDF file in the 'data' folder.")
        return

   
    assistant = RAGAssistant(pdf_path=pdf_file)
    assistant.build_pipeline()

    print("\n" + "="*50)
    print("Welcome to the Document Q&A Assistant!")
    print("Type 'exit' or 'quit' to leave.")
    print("="*50 + "\n")

    while True:
        try:
            query = input(" Enter your question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            result = assistant.ask(query)

            print("\n **Answer:**")
            print(result["answer"])

            print("\n **Source Page Citations (Source Page Citations):**")
            for c in result["citations"]:
                print(f" -  File: {c['file']} |  Page Number: {c['page']}")

            print("\n" + "-"*50 + "\n")

        except Exception as e:
            print(f" An error occurred: {e}")

if __name__ == "__main__":
    main()