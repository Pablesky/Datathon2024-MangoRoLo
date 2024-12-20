### Project Description: Design Attributes Logging Enhanced by AI

**Objective:**  
The buidl project aims to revolutionize clothing characterization by automating and standardizing the process through advanced AI technology. It leverages image analysis to identify, classify, and describe garments with precision while allowing for manual refinements to ensure accuracy and maintain user control.

For more information specifically about MLFLOW, go to the bottom of the README.

---


### About us:
This team is composed by Cristina Aguilera, Iker Garica, Pablo Vega and Sígrid Vila.


---


### Demo of the project:

<div style="text-align: center;">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=2CbmDZ9KB-0" target="_blank">
    <img src="http://img.youtube.com/vi/2CbmDZ9KB-0/0.jpg" 
         alt="Our demo!" width="400" height="340" border="10" />
  </a>
</div>

---


## Model Strucutre:
The model is based on the pretrained ResNet model, the last classification layer is eliminated and then some layers are added to bring the embedding size to the desired output size. Then, an MLP is added for each feature to predict the class of the attribute. These new layers are finetuned with the Mango Dataset and a weighted loss is used to mitigate the unbalancing of the dataset.

![Model Structure](Dibujo_sin_titulo.png)


---

### Key Features:

1. **AI-Driven Garment Analysis:**  
   - Utilizes computer vision and machine learning to analyze garment images.  
   - Identifies attributes such as fabric type, color, pattern, style, and specific design elements.  

2. **Standardization of Characterization:**  
   - Ensures consistent descriptions and classifications across datasets.  
   - Supports industry standards and integrates seamlessly with existing systems.  

3. **Manual Refinement Tools:**  
   - Offers users the ability to edit or refine AI-generated outputs.  
   - Balances automation with human expertise for optimal accuracy and contextual adjustments.  

4. **Customization & Adaptability:**  
   - Tailored to meet the needs of diverse stakeholders, including fashion designers, retailers, and sustainability initiatives.  
   - Flexible enough to incorporate new attributes or accommodate evolving industry requirements.  

5. **Data Integration:**  
   - Enables integration with databases or platforms for inventory management, trend analysis, and reporting.  
   - Offers export options in various formats for compatibility.  

---

### Benefits:

- **Efficiency:** Reduces the time and effort required for garment analysis.  
- **Consistency:** Standardizes the characterization process, minimizing discrepancies.  
- **Scalability:** Can handle large datasets, making it suitable for both small-scale and enterprise-level applications.  
- **User Empowerment:** Empowers users to retain control through manual adjustments, ensuring outputs meet specific needs.  

---

### Applications:

- **Retail & E-Commerce:** Automates product cataloging and improves customer search experiences.  
- **Fashion Design:** Aids designers in organizing and referencing collections.  
- **Sustainability Initiatives:** Helps assess and label garments for recycling or upcycling purposes.  
- **Data-Driven Insights:** Provides analytics on trends, popular styles, and consumer preferences.  

---

### Implementation Approach:

1. **Research & Development:**  
   - Develop and train AI models using diverse datasets of garment images.  
   - Test and refine algorithms to ensure high accuracy and relevance.

2. **User Interface Design:**  
   - Create an intuitive platform that balances automation with manual editing capabilities.  
   - Include features such as drag-and-drop uploads, detailed attribute lists, and visual previews.  

3. **Pilot Testing:**  
   - Collaborate with stakeholders for feedback and validation of the system in real-world scenarios.  
   - Address gaps and enhance functionalities based on user experiences.

4. **Deployment:**  
   - Launch the finalized platform with robust support and training for users.  
   - Provide ongoing updates to adapt to industry changes and user needs.  

---

### Folder structure 
The code is structured as follows:

        .
        ├── MangoData.py               # Dataset implementation to read csv and images of Mango.
        ├── MangoModel.py              # Model we developed for the task.
        ├── attributes.json            # Complimentary file for the StreamLit page.
        ├── parse_results.py           # Parsing of results from network to submission.
        ├── MangoRolo.py               # Main StreamLit page.
        ├── MangoGrad.py               # Code to compute the GradCam on images from the Mango Dataset.
        ├── Mango_presentation-pdf     # Presentation of the project.       
        └── train.py                   # Pipeline to train the model.
      
---
### Execution instructions:
To visualize the page, you need to run the streamlit2.py as follows:
```
streamlit run streamlit2.py
```


---
### Requirements:
- python 3.8
- conda
- pytorch 1.8.0
- torchvision 0.15.2
- cuda 10.2
- mlflow
- tqdm
- pandas
- numpy




### Future Vision:

The buidl project aspires to be a leading solution in AI-driven clothing characterization, setting benchmarks for innovation, efficiency, and user engagement. By streamlining processes and offering unparalleled customization, buidl will empower industries to achieve greater accuracy and sustainability in garment analysis.

---

## MLFlow:
We used this tool to keep track of the logs of the training process, specifically the loss. Moreover, we used another library to use GradCam to enhance the explainability of our model.
Down below, a visualization of the losses and some examples of the GradCam on clothes of the Mango dataset.

![Training Loss from MLFlow](train_loss.png)

<div align="center">
    <img src="gradcam.png" alt="GradCam example">
</div> <div align="center">
    <img src="gradcam1.png" alt="GradCam example">
</div>

