# Libre Scout
RAG based AI agent to scout over the internet and knowledge base for helping book buyers by providing informations.

### Technologies Used
- **SerpAPI**: For google search results
- **Salesforce BLIP Model**: For generating image captions.
- **FAISS Vector Database**: For storing the vector embeddings and similarity search
- **OpenAI GPT-3.5-turbo**: For generating responses to user questions.
- **Streamlit**: For building the interactive web interface.
- **Pillow**: For image processing.
- **Python Dotenv**: For managing environment variables.

## How to Run

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up your OpenAI API key**:
    - Create a `.env` file in the root directory of the project.
    - Add your OpenAI API key to the `.env` file:
      ```env
      OPENAI_API_KEY=your_openai_api_key
      SERPAPI_API_KEY=you_serpapi_key
      ```

5. **Run the Streamlit application**:
    ```sh
    streamlit run app.py
    ```

6. **Open the application**:
    - Open your web browser and go to `http://localhost:8501`.

## Usage

1. Upload an image in JPG, JPEG, or PNG format.
2. Wait for the image to be processed.
3. After processign you get the informations.


  
