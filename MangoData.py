from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
import os

diccionario_OHE = {
    "cane_height_type": ["invalid", "bloque", "cuña", "cuña abotinada", "alta", "baja", "media"],
    "closure_placement": ["invalid", "cuello", "sin cierre", "cierre delantero", "cierre trasero", "cierre hombro", "lateral"],
    "heel_shape_type": ["invalid", "plano", "bloque", "plataforma", "plataforma plana", "de aguja", "trompeta", "rectangular", "kitten", "embudo", "cuña", "plataforma en la parte delantera"],
    "knit_structure": ["invalid", "punto fino", "punto medio", "punto grueso", "unknown", "hecho a mano"],
    "length_type": ["invalid", "largo", "corto", "standard", "crop", "medio", "midi", "capri", "mini/micro", "asimétrico", "maxi", "tres cuartos", "tobillero"],
    "neck_lapel_type": ["hawaiano/bowling", "invalid", "capucha", "regular", "panadero", "cutaway", "caja", "pico", "mao", "smoking", "peak lapel", "alto/envolvente", "perkins", "button down", "halter", "escotado", "redondo", "polo", "camisero", "chimenea", "cisne", "off shoulder", "solapa", "cruzado", "shawl", "palabra honor", "babydoll/peter pan", "drapeado", "barca", "waterfall", "asimétrico", "espalda abierta", "kimono", "sin solapa"],
    "silhouette_type": ["regular", "invalid", "slim", "5 bolsillos", "jogger", "modern slim", "chino", "recto", "slouchy", "skinny", "acampanado/flare", "push up", "mom", "evase", "culotte", "palazzo", "acampanado/bootcut", "cargo", "boyfriend", "fino", "sarouel", "lápiz", "ancho", "oversize", "halter", "wide leg", "paperbag", "relaxed", "tapered", "bandeau", "superslim", "loose", "carrot", "parachute"],
    "sleeve_length_type": ["corta", "invalid", "larga", "tirante ancho", "tirante fino", "sin manga", "tres cuartos"],
    "toecap_type": ["invalid", "redonda", "con punta", "abierta", "cuadrada"],
    "waist_type": ["invalid", "ajustable/goma", "regular waist", "high waist", "low waist"],
    "woven_structure": ["invalid", "pesado", "ligero", "medio", "elástico"]
}

class MangoData(Dataset):
    def __init__(self, csv_path, height = None, width = None, use_models = True, img_folder = '', additional_transform = None) -> None:
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.diccionario_OHE = diccionario_OHE
        
        if additional_transform is not None:
            self.transform = additional_transform
        else:
            if height is None or width is None:
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((height, width)),
                    transforms.ToTensor()
                ])
            
        self.img_folder = img_folder
        
        if not use_models:
            # Get all rows where the column con_modelo is False
            self.df = self.df[self.df['con_modelo'] == False]
            
        # Drop the column con_modelo
        self.df = self.df.drop(columns = ['con_modelo'])
            
    def __len__(self):
        return len(self.df)
    
    def load_image(self, path: int) -> Image.Image:
        "Opens an image via a path and returns it."
        return Image.open(path)
    
    def one_hot_encode(self, column_name, value):
        """One-hot encodes a value based on the diccionario_OHE."""
        categories = self.diccionario_OHE.get(column_name, [])
        encoding = [1 if value == category else 0 for category in categories]
        return encoding
    
    def __getitem__(self, index):
        # Get the image path
        image_path = self.df.iloc[index]['des_filename']
        # Open the image
        image_path = os.path.join(self.img_folder, image_path)
        image = self.load_image(image_path)
        image = self.transform(image)
        
        encoded_features = []
        for column in self.diccionario_OHE.keys():
            value = self.df.iloc[index].get(column, 'invalid')  # Default to 'invalid' if not found
            encoded_features.append(torch.tensor(self.one_hot_encode(column, value), dtype=torch.long))
        
        # image = torch.tensor(image, dtype=torch.float32)
        image = image.to(dtype=torch.float32)
        
        return image, encoded_features
            
if __name__ == '__main__':
    images_path = os.path.join('images', 'images')
    mango_data = MangoData(
        csv_path = 'train.csv', 
        use_models=True,
        img_folder = images_path)
    
    # Verify that the images are loaded correctly
    img, label = mango_data[0]
    
    print(len(label))
            
    