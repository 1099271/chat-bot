from graph import graph


def stream_graph_updates(user_input: str):
    # user_input = "I need some expert guidance for building an AI Agent. Could you request assistance for me?"
    events = graph.stream(
        {
            "messages": [{"role": "user", "content": user_input}],
            "name": "",  # 初始化 name 字段
            "birthday": "",  # 初始化 birthday 字段
        },
        config={"configurable": {"thread_id": "1"}},
        stream_mode="values",
    )

    for event in events:
        print(event)
        if "messages" in event:
            event["messages"][-1].pretty_print()
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)


# print(tool.invoke("What is the latest news about LangGraph?"))
# raise SystemExit

user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
)
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("Goodbye!")
#             break
#         stream_graph_updates(user_input)
#     except KeyboardInterrupt:
#         print("\nGoodbye!")
#     except:
#         user_input = "What do you know about LangGraph?"
#         print("User:" + user_input)
#         stream_graph_updates(user_input)
#         break
