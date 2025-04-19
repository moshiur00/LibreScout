from openai import OpenAI

client = OpenAI()

def build_prompt(extracted_text, question=None):
    base_prompt = f"""
        Context: The following text is extracted from a book page via OCR. It may be incomplete or have minor errors.
        Extracted Text: \"\"\" {extracted_text} \"\"\"
    """

    if question:
        base_prompt += f"""Question: {question}
    
    Instruction: Answer based only on the provided text. If the answer is not clearly available, state that the information is incomplete. """
    else:
        base_prompt += """Instruction: Summarize the key information from the text with book name, author, publication year and short summary review. Do not assume missing information. """
    
    return base_prompt.strip()


def generate_response(retrieved_texts, query, max_tokens=150):
    
    context = "\n".join(retrieved_texts)
    prompt = build_prompt(context, query)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1
    )
    return response.choices[0].message.content

