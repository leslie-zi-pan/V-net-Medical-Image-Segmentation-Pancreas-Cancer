# V-net-Medical-Image-Segmentation-Pancreas-Cancer
Custom V-net architecture for Medical Image segmentation of the pancreas and pancreatic tumours from CT scans. 

Data not uploaded due to size and confidentiality. 

The data consisted of 281 training images, with its respective labels, and 129 testing data
Trained up to ~340 Epochs. 

Pre-augmented data training results: 

![image](https://user-images.githubusercontent.com/43177212/114284708-c860a880-9a49-11eb-8ff0-969e2de351f5.png)
![image](https://user-images.githubusercontent.com/43177212/114284709-ca2a6c00-9a49-11eb-80c8-ba952d56ba77.png)


Post aumentation training results(elastic and affine): 

![image](https://user-images.githubusercontent.com/43177212/114284639-2345d000-9a49-11eb-9506-df25942c6681.png)
![image](https://user-images.githubusercontent.com/43177212/114284647-2b9e0b00-9a49-11eb-9ac4-cf3c59611f1c.png)

Hyper-parameter tuning graphs: 

![image](https://user-images.githubusercontent.com/43177212/114284649-32c51900-9a49-11eb-942c-bdf40b715514.png)
![image](https://user-images.githubusercontent.com/43177212/114284653-36f13680-9a49-11eb-834d-693cecf6798c.png)

Example Patch based training results:

![image](https://user-images.githubusercontent.com/43177212/114284656-407a9e80-9a49-11eb-85d7-7adbaddecaac.png)

Whole image results not as good as epected. Patch based training used due to resource limitations. Improvements to results include use of 2D training with spatial recognition for better resource optimization and preventing patchy predictions on whole image. 

References: 
F. Milletari, N. Navab, and S. A. Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” Proc. - 2016 4th Int. Conf. 3D Vision, 3DV 2016, pp. 565–571, 2016, doi: 10.1109/3DV.2016.79.
