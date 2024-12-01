import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def classify_image(img_path):
    """
    A function to classify an image to one of the following categories:
    ['Advertisement', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']

    :param img_path: path to the image
    :return: category of the image
    """

    api = "AIzaSyDUFYHbrO4XG_Yb3Aq_IEDLOJKCaqWAu9s"
    # Encode the image in base64
    with open(img_path, "rb") as img_file:
        # Encode the image in base64
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        # Create the base64 string in the expected format
        img_b64_str = f"data:image/png;base64,{img_base64}"

    model_name = "gemini-1.5-flash"
    llm = ChatGoogleGenerativeAI(api_key=api, model=model_name, temperature=0.8)
    prompt = """
        classify this document to one of these categories:
['ADVE',

 'Email',

 'Form',

 'Letter',

 'Memo',

 'News',

 'Note',

 'Report',

 'Resume',

 'Scientific']
        """
    # Pass the image and the prompt to the model
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": img_b64_str}
        ]
    )

    response = llm.invoke([message])
    return response.content