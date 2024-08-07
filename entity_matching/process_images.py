import json
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from deepface import DeepFace

def load_entities(json_path):
    """
    Load JSON entities from a file.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    entities = [{'name': key, 'embedding': value['embedding']} for key, value in data.items()]
    return entities

def parse_image(image_path):
    """
    Performs entity matching on the given image 
    """
    img = cv2.imread(image_path)
    try:
        # Using 'represent' to get the face embeddings directly
        embedding = DeepFace.represent(img, enforce_detection=False)
        return embedding if embedding else []
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return []


def entity_grounding(faces, entities, threshold=0.2):
    """
    Check for entity grounding by matching recognized faces with important entities.
    """
    grounded_entities = []
    for face in faces:
        face_embedding = face['embedding'] 
        for entity in entities:
            entity_embedding = np.array(entity['embedding'])
            similarity = 1 - cosine(face_embedding, entity_embedding)
            print(similarity)
            if similarity >= threshold:
                grounded_entities.append({
                    'name': entity['name'],
                    'similarity': similarity,
                    'embedding': entity['embedding']
                })
    return grounded_entities

def craft_prompt(grounded_entities):
    """
    Crafts prompts given the grounded entities from the image
    """
    if not grounded_entities:
        return "no entities found in the image."

    prompt = ''
    for entity in grounded_entities:
        prompt += f"Who is {entity['name']} and what are the events surrounding him?\n"

    return prompt

def run(image_path, json_path):
    entities = load_entities(json_path)
    faces = parse_image(image_path)
    grounded_entities = entity_grounding(faces, entities)
    query_prompt = craft_prompt(grounded_entities)
    print(query_prompt)
    return query_prompt
