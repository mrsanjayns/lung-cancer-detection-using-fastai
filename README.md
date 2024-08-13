# lung-cancer-detection using fast-ai api 

dataset: https://arxiv.org/abs/1912.12142 
This is the dataset which I have used in the project, It contains **histopathological images** of lung cancer, divided into 3 classes.
The classes I have choosen are lung_aca, lung_scc, and benign. If this link is broken in the near future please search for LC25000 dataset, it has both the lung and colon cancer images.

Next step is splitting the dataset into train, test and validation. 
I have used split_folders to do so. If you want to learn the same take a look at this video https://youtu.be/C6wbr1jJvVs?si=BFOfXTUjRv8HfLCU

Now that dataset is splitted into test, train and val. 
Develop a model using this colab file https://colab.research.google.com/drive/1_znUNp5k5bL5AXGoLJJ4A4DibxyUWV7q?usp=sharing
This code uses **resnet34 architecture**.

Atlast clone this repo or download it as zip. Download a saved model from colab. make changes into app.py file for dataset and pretrained model path according to your system path for dataset, and pretrained pytorch model.

now move to the project directory.
open terminal or cmd 
type _python3 app.py_ hit enter 

Output images are as follows. 
![image](https://github.com/user-attachments/assets/06f69e69-a4d8-4c66-9720-e61416605e1b)
<img width="1421" alt="image" src="https://github.com/user-attachments/assets/bf2bb836-6cc9-40fa-9189-856d9dadf6f7">





