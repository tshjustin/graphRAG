import json
from deepface import DeepFace
import cv2
import numpy as np 
from scipy.spatial.distance import cosine 

def load_entities(json_path):
    """
    Load  JSON 
    """
    with open(json_path, 'r') as file:
        entities = json.load(file)
    return entities['entities']

def parse_image(image_path):
    """
    Performs entity matching on the given image using DeepFace for facial recognition
    """
    img = cv2.imread(image_path)
    
    try:
        faces = DeepFace.analyze(img, actions=['age', 'gender', 'emotion'])
    except Exception as e:
        print(f"Error in analyzing image: {e}")
        faces = []

    return faces

def entity_grounding(faces, entities, threshold=0.4):
    """
    Check for entity grounding by matching recognized faces with important entities.
    """
    grounded_entities = []
    for face in faces:
        face_embedding = face['embedding']
        for entity in entities:
            entity_embedding = np.array(entity['embedding'])
            similarity = 1 - cosine(face_embedding, entity_embedding)
            if similarity >= threshold:
                grounded_entities.append({
                    'name': entity['name'],
                    'similarity': similarity
                })
    return grounded_entities

def craft_prompt(grounded_entities, caption):
    """
    Crafts prompts given the grounded entities from the image and the provided caption.
    """
    if not grounded_entities:
        return "No recognizable entities found in the image."

    prompt = f"Image Caption: {caption}\n"
    prompt += "Querying graph with the following entities:\n"
    
    for idx, entity in enumerate(grounded_entities):
        name = entity['name']
        similarity = entity['similarity']
        embedding_str = ', '.join([f"{val:.2f}" for val in entity['embedding']])
        prompt += f"{idx+1}. {name} with a similarity score of {similarity:.2f}.\n"
        prompt += f"Embedding: [{embedding_str}]\n"

    if similarity < 0.5: 
        

    return prompt


def main(image_path, json_path):
    entities = load_entities(json_path)
    faces = parse_image(image_path)
    grounded_entities = entity_grounding(faces, entities)
    query_prompt = craft_prompt(grounded_entities)
    print(query_prompt)

if __name__ == "__main__":
    image_path = "/home/tjustin/ragtest/data/news_PE_subset_images/TODAY_2241206_img2.jpg"
    json_path = "/home/tjustin/Projects/Graph-RAG-MS/entity_matching/face_entity.json"
    main(image_path, json_path)
