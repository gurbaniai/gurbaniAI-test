"""GurbaniAI CLI agent — interactive chat about Sikh scriptures."""

from gurbani_agent import ask_gurbani_llm, get_provider_info


def main() -> None:
    provider, model = get_provider_info()
    messages: list[dict[str, str]] = []

    print("GurbaniAI")
    print("Ask about meanings and teachings in Sikh scriptures.")
    print(f"Provider: {provider} | Model: {model}")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            user_prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nWaheguru Ji Ka Khalsa, Waheguru Ji Ki Fateh.")
            break

        if not user_prompt:
            continue
        if user_prompt.lower() in {"quit", "exit"}:
            print("Waheguru Ji Ka Khalsa, Waheguru Ji Ki Fateh.")
            break

        messages.append({"role": "user", "content": user_prompt})
        print("Reflecting on Gurbani...")
        answer = ask_gurbani_llm(messages)
        print(f"\nGurbaniAI: {answer}\n")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
