from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
import logging
import json

### Variables

# Init dynamoDB client
dynamodb_client = boto3.client('dynamodb')

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logger = logging.getLogger(__name__)

# S3 Configuration
BUCKET_NAME = "andrew-perkins-in-usa-ohio"
BUCKET_KEY = "ImageAnalysis/image"

# Bedrock Configuration
bedrock_client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

### Functions

def fetch_rekognition_results_from_dynamoDB():
    response = dynamodb_client.get_item(
        TableName='ImageAnalysisResults',
        Key={
            'ImageKey': {'S': 'ImageAnalysis/image'},
        }
    )
    print(response['Item'])
    return response.get('Item', {}).get('Rekognition', {})

def call_bedrock_llm(question, rekognition_data):

    # Prepare the prompt for the LLM
    prompt = f"""
    The following are results from image analysis: {rekognition_data}.
    Question: {question}
    Please provide a concise, human-readable answer based on the data above.
    """

    print(prompt)

    # Prepare the request body
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 200,
            "temperature": 0.1,
            "topP": 0.9,
            "stopSequences": []
        }
    })

    # Bedrock model configuration
    modelId = "amazon.titan-text-express-v1"  # Titan Text G1 - Express model
    accept = "application/json"
    contentType = "application/json"

    # Create the Bedrock client
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    try:
        # Call the Bedrock model
        response = bedrock_client.invoke_model(
            body=body,
            modelId=modelId,
            accept=accept,
            contentType=contentType
        )

        # Parse the response
        response_body = json.loads(response["body"].read())
        llm_output = response_body.get("results")[0].get("outputText")

        return llm_output

    except Exception as e:
        print(f"Error calling Bedrock: {str(e)}")
        return None

def call_bedrock_llm_old(question, rekognition_data):
    # Prepare the payload for the LLM
    prompt = f"""
    The following are results from image analysis: {rekognition_data}.
    Question: {question}
    Please provide a concise, human-readable answer based on the data above.
    """

    print(prompt)

    body = json.dumps({
        "prompt": prompt,
        "maxTokenCount": 200,
        "temperature": 0.1,
        "top_p": 0.9,
    })

    modelId = 'amazon.titan-text-express-v1' # Titan Text G1 - Express model
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    response_body = json.loads(response.get('body').read())
    llm_output = response_body.get('completion')

    # Parse and return the LLM response
    return llm_output

### Routes

@app.route("/")
def health_check():
    return jsonify({"status": "Flask server is running - image-upload-and-process"}), 200

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    logging.warning(f"Route /upload")

    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file in request")
        return jsonify({"error": "No selected file"}), 400

    # Upload to S3
    s3 = boto3.client('s3', region_name='us-east-2')
    try:
        # Upload the file to the bucket
        s3.upload_fileobj(file, BUCKET_NAME, BUCKET_KEY)

        # Generate a pre-signed URL to access the uploaded image
        image_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': BUCKET_KEY},
            ExpiresIn=3600  # URL valid for 1 hour
        )

        print(f"File uploaded successfully. URL: {image_url}")
        return jsonify({"image_url": image_url}), 200

    except Exception as e:
        print(f"Error during file upload: {e}")
        return jsonify({"error": str(e)}), 500

# Route to handle the question submission
@app.route('/ask', methods=['POST'])
def ask_question():

    print(f"Route /ask")
    data = request.get_json()

    if not data:
        logging.error(f"Error 'data' not defined in call: {e}")
        return jsonify({"error": "Invalid JSON in request"}), 400

    print(f"Received request: {data}")

    question = data.get("question")
    image_url = data.get("image_url")
    if not question or not image_url:
        logging.error(f"Error 'data' does not contain both 'question' and 'image_url': {e}")
        return jsonify({"error": "Both 'question' and 'image_url' are required"}), 400

    # Get rekognition result from analysing the image
    try:
        rekognition_results = fetch_rekognition_results_from_dynamoDB()
    except Exception as e:
        logging.error(f"Error calling DynamoDB: {e}")
        return jsonify({"error": "Failed to get data from DynamoDB"}), 500

    # Call Bedrock LLM
    try:
        answer = call_bedrock_llm(question, rekognition_results)
    except Exception as e:
        logging.error(f"Error calling Bedrock: {e}")
        return jsonify({"error": "Failed to generate answer using LLM"}), 500

    # Return answer to the question
    return jsonify({
        "question": question,
        "image_url": image_url,
        "answer": answer
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

