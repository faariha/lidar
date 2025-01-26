import os
import open3d as o3d
import fitz  # PyMuPDF for reading PDFs
from flask import Flask, request, jsonify
from flask_cors import CORS
import ast  # For parsing string representations of lists
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage, firestore
import re
from pypdf import PdfReader
import shutil
import uuid  # For generating random names

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"emergency-help-c6266-firebase-adminsdk-mq841-99fe6e3010.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "emergency-help-c6266.appspot.com"
})

# Initialize Firestore
db = firestore.client()

app = Flask(__name__)
CORS(app)

# Directories for uploads and output
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def generate_random_filename(extension=".glb"):
    """
    Generate a random text-based filename using a UUID.
    """
    random_name = uuid.uuid4().hex[:10]  # Generate a 10-character random string
    return f"model_{random_name}{extension}"

def save_model_reference_to_firestore(collection_name, document_id, data):
    """
    Saves a reference to a 3D model in Firebase Firestore.
    Args:
        collection_name (str): The name of the Firestore collection.
        document_id (str): The unique ID of the document.
        data (dict): The data to store in the document.
    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    try:
        db.collection(collection_name).document(document_id).set(data)
        print(f"Model reference saved in Firestore under collection '{collection_name}' with document ID '{document_id}'.")
        return True
    except Exception as e:
        print(f"Error saving reference to Firestore: {e}")
        return False

# def extract_points_from_pdf(pdf_path):
#     """
#     Extracts point data from a PDF file where points are formatted as:
#     [x, y, z]
#     """
#     points = []
#     try:
#         with fitz.open(pdf_path) as pdf:
#             for page in pdf:
#                 text = page.get_text()
#                 for line in text.splitlines():
#                     try:
#                         point = ast.literal_eval(line.strip())
#                         if isinstance(point, list) and len(point) == 3:
#                             points.append(point)
#                     except (ValueError, SyntaxError):
#                         continue
#     except Exception as e:
#         print(f"Error reading PDF: {e}")
#         return None
#     return points

# def normalize_points(points):
#     """
#     Normalize points by centering them and scaling to fit within a unit sphere.
#     """
#     points = np.array(points)
#     centroid = np.mean(points, axis=0)
#     points -= centroid
#     max_distance = np.max(np.linalg.norm(points, axis=1))
#     points /= max_distance
#     return points.tolist()

def pdf_to_mesh(filepath, output_path):
    """
    Process PDF and generate a 3D model (GLB format).
    """
    try:
        points = []
        pattern = r"-?\d+\.\d+"
        pdf = PdfReader(filepath)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        values = re.findall(pattern, text)

        point3 = []
        for value in values:
            point3.append(float(value))
            if len(point3) == 3:
                points.append(point3)
                point3 = []

        points_pcd = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_pcd)
        temp_ply = "temp.ply"
        o3d.io.write_point_cloud(temp_ply, pcd)

        pcd = o3d.io.read_point_cloud(temp_ply)
        alpha = 0.0011
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        mesh.compute_convex_hull()
        
        mesh_out = mesh.filter_smooth_simple(number_of_iterations=6)

        o3d.io.write_triangle_mesh(output_path, mesh_out)
        os.remove(temp_ply)
        return output_path
    except Exception as e:
        print(f"Error processing PDF to GLB: {str(e)}")
        return None

def upload_to_firebase(file_path, destination_blob_name):
    """
    Uploads a file to Firebase Storage and returns the public URL.
    """
    try:
        bucket = storage.bucket()
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        blob.make_public()
        public_url = blob.public_url
        print(f"File uploaded to Firebase: {public_url}")
        return public_url
    except Exception as e:
        print(f"Error uploading file to Firebase Storage: {str(e)}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are accepted'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Generate a random filename for the output GLB file
    output_filename = generate_random_filename()
    output_model = pdf_to_mesh(file_path, os.path.join(OUTPUT_FOLDER, output_filename))

    if not output_model:
        return jsonify({'error': 'Failed to generate 3D model'}), 500

    firebase_url = upload_to_firebase(output_model, f"3d_models/{output_filename}")

    if not firebase_url:
        return jsonify({'error': 'Failed to upload 3D model to Firebase'}), 500

    model_data = {
        'file_name': output_filename,
        'model_url': firebase_url,
        'uploaded_at': firestore.SERVER_TIMESTAMP
    }
    collection_name = "3D_Models"
    document_id = output_filename.split(".")[0]
    if not save_model_reference_to_firestore(collection_name, document_id, model_data):
        return jsonify({'error': 'Failed to save model reference in Firestore'}), 500

    return jsonify({
        'message': '3D model generated, uploaded, and reference saved successfully',
        'model_url': firebase_url
    })

if __name__ == "__main__":
    app.run(debug=True)
