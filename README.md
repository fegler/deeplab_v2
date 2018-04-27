# deeplab_v2
deeplab v2 implementation using matconvnet


Download pre-trained model & dataset  
http://liangchiehchen.com/projects/DeepLab.html

### citation  
@article{CP2016Deeplab,
  title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs},
  author={Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille},
  journal={arXiv:1606.00915},
  year={2016}
}

### reference  
https://github.com/vlfeat/matconvnet-fcn -> fcn matconvnet implement  
https://github.com/albanie/mcnDeepLab -> deeplab v2 matconvnet training code is yet.  
https://github.com/xmyqsh/deeplab-v2 -> caffe implementation.  

### prerequisition
matconvnet1.0-beta25 -> http://www.vlfeat.org/matconvnet/  


using pascal VOC2012 dataset  
result (only epoch 40 and only train/val)   
not sufficient result yet.  
  
### Screenshot  
objective is loss calculate layer ouput name.  

![error rate result](https://user-images.githubusercontent.com/30647846/39246752-7184852a-48d2-11e8-8f37-02af37451051.png)

