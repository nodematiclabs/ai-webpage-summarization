import requests
import tiktoken

from bs4 import BeautifulSoup
from vertexai.language_models import ChatModel

chat_model = ChatModel.from_pretrained("chat-bison@001")
parameters = {"temperature": 0.2, "max_output_tokens": 128, "top_p": 0.8, "top_k": 40}
encoding = tiktoken.encoding_for_model("gpt-4")

URL = "https://example.com/"

def extract_text(url):
    # Send a GET request to the webpage
    response = requests.get(url)
    
    # If the GET request is successful, the status code will be 200
    if response.status_code == 200:
        # Get the content of the response
        webpage = response.text
        
        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(webpage, "html.parser")
        
        # Get the text of the webpage
        text = soup.get_text()
        
        return text
    else:
        raise Exception("Failed to retrieve webpage.")

text = extract_text(URL)
message = f"Write a very short, one-sentence SEO description for a website containing the following text:\n\n{text}"
while len(encoding.encode(message)) > 8000:
    message = " ".join(message.split(" ")[0:-1])

chat = chat_model.start_chat(
    examples=[]
)
response = chat.send_message(message, **parameters)
print(response.text)