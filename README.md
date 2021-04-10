# V-net-Medical-Image-Segmentation-Pancreas-Cancer
Custom V-net architecture for Medical Image segmentation of the pancreas and pancreatic tumours from CT scans. 

Data not uploaded due to size and confidentiality. 

The data consisted of 281 training images, with its respective labels, and 129 testing data
Trained up to ~340 Epochs. 

Pre-augmented data training results: 
![image](https://user-images.githubusercontent.com/43177212/114284513-63f11980-9a48-11eb-96c8-44d2af3fd7b7.png)
![image](https://user-images.githubusercontent.com/43177212/114284515-65224680-9a48-11eb-8b8e-9d24badd5f46.png)

Post aumentation training results(elastic and affine): 
![image](https://user-images.githubusercontent.com/43177212/114284503-52a80d00-9a48-11eb-958d-2a82e17cff09.png)
![image](https://user-images.githubusercontent.com/43177212/114284504-550a6700-9a48-11eb-858f-841a0e99b897.png)

Hyper-parameter tuning graphs: 
![image](https://user-images.githubusercontent.com/43177212/114284529-83884200-9a48-11eb-92f9-785510e683e4.png)
![image](https://user-images.githubusercontent.com/43177212/114284532-85520580-9a48-11eb-9b8d-d267218e3e4d.png)

Example Patch based training results:
![image](https://user-images.githubusercontent.com/43177212/114284542-90a53100-9a48-11eb-9b15-08f75609ef43.png)

Whole image results not as good as epected. Patch based training used due to resource limitations. Improvements to results include use of 2D training with spatial recognition for better resource optimization and preventing patchy predictions on whole image. 

References: 
F. Milletari, N. Navab, and S. A. Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” Proc. - 2016 4th Int. Conf. 3D Vision, 3DV 2016, pp. 565–571, 2016, doi: 10.1109/3DV.2016.79.
