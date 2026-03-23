client<llm> GPT4oMini {
  provider "openai"
  options {
    model "gpt-4o-mini"
    temperature 0.1
    api_key env.OPENAI_API_KEY
  }
}

client<llm> GPT4o {
  provider "openai"
  options {
    model "gpt-4o"
    temperature 0.1
    api_key env.OPENAI_API_KEY
  }
}
