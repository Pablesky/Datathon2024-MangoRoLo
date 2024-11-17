import pandas as pd
from PIL import Image
from torchvision import transforms
import os
from MangoModel2 import MangoModel
import torch
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

translation = {
    "cane_height_type": ["INVALID", "Bloque", "Cuña", "Cuña abotinada", "Alta", "Baja", "Media"],
    "closure_placement": ["INVALID", "Cuello", "Sin cierre", "cierre delantero", "cierre trasero", "cierre hombro", "lateral"],
    "heel_shape_type": ["INVALID", "Plano", "Bloque", "Plataforma", "Plataforma plana", "De aguja", "Trompeta", "Rectangular", "Kitten", "Embudo", "Cuña", "Plataforma en la parte delantera"],
    "knit_structure": ["INVALID", "Punto Fino", "Punto Medio", "Punto Grueso", "UNKNOWN", "Hecho a mano"],
    "length_type": ["INVALID", "Largo", "Corto", "Standard", "Crop", "Medio", "Midi", "Capri", "Mini/Micro", "Asimétrico", "Maxi", "Tres Cuartos", "Tobillero"],
    "neck_lapel_type": ["Hawaiano/Bowling", "INVALID", "Capucha", "Regular", "Panadero", "Cutaway", "Caja", "Pico", "Mao", "Smoking", "Peak Lapel", "Alto/Envolvente", "Perkins", "Button Down", "Halter", "Escotado", "Redondo", "Polo", "Camisero", "Chimenea", "Cisne", "Off Shoulder", "Solapa", "Cruzado", "Shawl", "Palabra Honor", "Babydoll/Peter Pan", "Drapeado", "Barca", "Waterfall", "Asimétrico", "Espalda Abierta", "Kimono", "Sin solapa"],
    "silhouette_type": ["Regular", "INVALID", "Slim", "5 Bolsillos", "Jogger", "Modern slim", "Chino", "Recto", "Slouchy", "Skinny", "Acampanado/Flare", "Push Up", "Mom", "Evase", "Culotte", "Palazzo", "Acampanado/Bootcut", "Cargo", "Boyfriend", "Fino", "Sarouel", "Lápiz", "Ancho", "Oversize", "Halter", "Wide leg", "Paperbag", "Relaxed", "Tapered", "Bandeau", "Superslim", "Loose", "Carrot", "Parachute"],
    "sleeve_length_type": ["Corta", "INVALID", "Larga", "Tirante Ancho", "Tirante Fino", "Sin Manga", "Tres Cuartos"],
    "toecap_type": ["INVALID", "Redonda", "Con punta", "Abierta", "Cuadrada"],
    "waist_type": ["INVALID", "Ajustable/Goma", "Regular Waist", "High Waist", "Low Waist"],
    "woven_structure": ["INVALID", "Pesado", "Ligero", "Medio", "Elástico"]}


numClasses = [7, 7, 12, 6, 13, 34, 34, 7, 5, 5, 5]

# Data Preparation
mangoTransforms = transforms.Compose([
    transforms.Resize(224),               # Resize the shorter side to 224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def main():
    mangoModel = MangoModel(num_classes_list=numClasses).to(device)
    mangoModel.load_state_dict(torch.load('new_weights/mango_model_6.pth'))
    mangoModel.eval()
    
    df = pd.read_csv('test_data.csv')
    df = df.drop_duplicates(subset=df.columns[0], keep='first')
    
    files = df['des_filename'].unique()
    img_path = 'images/images'
        
    ids = []
    results = []
    
    for file in files:
        try:
            img = Image.open(os.path.join(img_path, file))
            img_input = mangoTransforms(img)
            id_img = "_".join(file.split('_')[:2])
            
            img_input = img_input.unsqueeze(0).to(device)
            output_logits = mangoModel(img_input)
            
            probabilities = [softmax(out, dim=1).cpu().detach().numpy().squeeze() for out in output_logits]
            
            for i, (key, value) in enumerate(translation.items()):
                id_unique = f"{id_img}_{key}"
                ids.append(id_unique)
                
                index_best = probabilities[i].argmax().item()
                result = value[index_best]
                results.append(result)
        except:
            id_img = "_".join(file.split('_')[:2])
            for i, (key, value) in enumerate(translation.items()):
                id_unique = f"{id_img}_{key}"
                ids.append(id_unique)
                
                results.append('INVALID')
            
    submission = pd.DataFrame({'test_id': ids, 'des_value': results})
    submission.to_csv('submission.csv', index=False)   
            
if __name__ == '__main__':
    main()