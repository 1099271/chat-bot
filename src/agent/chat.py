from graph import graph


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# print(tool.invoke("What is the latest news about LangGraph?"))
# raise SystemExit


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except:
        user_input = "What do you know about LangGraph?"
        print("User:" + user_input)
        stream_graph_updates(user_input)
        break
