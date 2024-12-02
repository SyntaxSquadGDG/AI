from fastapi import FastAPI, Query, HTTPException, Request
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import re
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Image Classification API",
    description="API to classify images into predefined categories.",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ccdtr14p-3000.uks1.devtunnels.ms",
    "https://syntaxsquad-ai.azurewebsites.net"
]

# config Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def classify_image_bytes(image_bytes: bytes) -> str:
    """
    Classify an image to one of the specified categories.

    :param image_bytes: Image data in bytes.
    :return: Category of the image.
    """
    api = "AIzaSyDUFYHbrO4XG_Yb3Aq_IEDLOJKCaqWAu9s"
    if not api:
        raise ValueError("Google API key not found in environment variables.")

    # Encode the image in base64
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    # Create the base64 string in the expected format
    img_b64_str = f"data:image/png;base64,{img_base64}"

    model_name = "gemini-1.5-flash"
    llm = ChatGoogleGenerativeAI(api_key=api, model=model_name, temperature=0.8)
    prompt = """
        classify this document to one of these categories:
['Advertisement',
 'Email',
 'Form',
 'Letter',
 'Memo',
 'News',
 'Note',
 'Report',
 'Resume',
 'Scientific']
 
 And give me the accuracy of your decision from 0 "not sure" to 100 "Sure".
 Output Format:
 Email 87
    """
    # Pass the image and the prompt to the model
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": img_b64_str}
        ]
    )

    response = llm.invoke([message])
    response_content = str(response.content)
    match = re.match(r"(\w+)\s+(\d+)", response_content)
    folder = match.group(1)
    accuracy = int(match.group(2))
    section = "/SyntaxSquad/"
    result = {
        "path": section + folder,
        "accuracy": accuracy
    }
    return section + folder, accuracy

@app.get("/")
def hello_world():
    return {"message":"Hello World! CORS 3"}
@app.post("/classify-image/")
async def classify_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded image.

    :param file: Image file uploaded by the user.
    :return: JSON response with the classification result.
    """
    try:
        # Validate the uploaded file's content type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid image type. Only PNG and JPEG are supported.")

        # Read the image bytes
        image_bytes = await file.read()

        # Check the size of the image (limit to 5MB)
        if len(image_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image size exceeds 5MB limit.")

        # Classify the image
        path, accuracy = classify_image_bytes(image_bytes)

        return JSONResponse(content={"path": path, "accuracy": accuracy})

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        # Log the exception details if necessary
        raise HTTPException(status_code=500, detail="An error occurred while processing the image.")

if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
