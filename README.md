# quantum_soil_erosion
Soil erosion detection task for Quantum.
Due to github limitations, JP2 file is not included.

## Files

data_prep.ipynb - Discovering data with https://medium.datadriveninvestor.com/preparing-aerial-imagery-for-crop-classification-ce05d3601c68 guide. Making a dataset of features and masks images.

main.py - data loading and model training.

## Model

As far as, aim of model is to mark soil erosion places, I decided to use an autoencoder, that will produce a binary mask for given area pictures.

# Conclusion
To begin with, soil erosion is a natural process that happens all the time. But, development of agriultural techniques significantly speeded up this process. Most common reasons of soil erosion in Ukraine:
- Deforestation
- No crop rotation
- Wrong hills crop placement  

### To prevent soil erosion, people should:
* Use agroforestry. By doing this, wind will have less erosion impact on soil.
* Plant crops in special patterns, so water runoff will be slowed down.
* Use crop rotation. As a result, different crops will not have such destructive impact on soil.
* Planting crops on hills on different levels. This technique is widely spread in hilly areas, Ukraine have plenty of them.
