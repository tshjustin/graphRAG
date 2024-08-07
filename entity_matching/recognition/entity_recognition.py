from deepface import DeepFace
from PIL import Image, ImageDraw
import os 
import json 

# Poor image quality 
# george_goh_img = "/home/tjustin/ragtest/data/news_PE_subset_images/CNA_3664946_img1.jpg"
# tharman_img = "/home/tjustin/ragtest/data/news_PE_subset_images/TODAY_2241206_img2.jpg"

'''Creates and saves important entities from sample corpus'''

ng_kok_song_img = "/home/tjustin/ragtest/data/news_PE_subset_images/CNA_3642366_img1.jpg"

tan_kin_lian_img = "/home/tjustin/ragtest/data/news_PE_subset_images/TODAY_2231246_img1.jpg"

trio = "/home/tjustin/ragtest/data/news_PE_subset_images/TODAY_2243166_img1.jpg"

images = [trio]

def extract_faces(img_path):
  '''Use VGG by default to extract relevant entities'''
  embedding_objs = DeepFace.represent(img_path = img_path, enforce_detection=False)

  for embedding in embedding_objs:
    print(embedding['facial_area'], embedding['face_confidence'])
  
  return embedding_objs

def draw_bounding_box(image_path: str, embedding_body):
  '''
  Draw bounding box on detected faces

  Returns the mapping of colored boxes to recognized personnel (Manual Labelling) 
  '''
  image = Image.open(image_path)
  draw = ImageDraw.Draw(image)

  colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan']
  color_embedding_map = {} # perform a mapping of color boxes to entity 

  for i, embedding in enumerate(embedding_body):
    x = embedding['facial_area']['x']
    y = embedding['facial_area']['y']
    w = embedding['facial_area']['w']
    h = embedding['facial_area']['h']
    color = colors[i % len(colors)]  
    draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
    color_embedding_map[color] = embedding

  image_name = os.path.basename(image_path)
  output_path = f"/home/tjustin/deepface/bounding_box_images/{image_name}"
  image.save(output_path)

  return color_embedding_map 

def save_entity_embeddings(personnel_embedding_map, output_file):
    with open(output_file, 'w') as f:
        json.dump(personnel_embedding_map, f, indent=4)

if __name__ == "__main__":
  output_file = "/home/tjustin/deepface/graph-rag/face_entity.json"

  for image in images: 
    print(f"Exracting and drawing for {image}")
    faces_embeddings = extract_faces(image)
    color_embedding_map = draw_bounding_box(image, faces_embeddings)

    save_entity_embeddings(color_embedding_map, output_file)

    # # print(color_embedding_map)
    # for color, embedding in color_embedding_map.items():
    #   print(f"Color: {color}, Embedding:")

    