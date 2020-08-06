Dependencies:
	
	Python 3.7 

	Opencv-conrtib-python 4.3.0.36 

	Numpy 1.16.2 

	Scikit-learn 0.21.2 

	Tensorflow 2.2.0 

	Keras 2.3.1

Run Specification:
	
	For dataset "Fluo-N2DL-HeLa" and "PhC-C2DL-PSC", run these code:
		 
		FluoTask1.py,
		FluoTask2.py
		FluoTask3.py
		PhcTask1.py
		PhcTask2.py
		PhcTask3.py

	The Dataset path variable “path” defined in "main" function and default value is root path, if the path is different, you need to change the variable manually.
	
	For dataset "DIC-C2DH-HeLa" we use "deepwater" model which is a open source model to predict the image mask area. 

	The deepwate model run method is here:
	https://gitlab.fi.muni.cz/xlux/deepwater/-/blob/master/README.md
	
	Then "DICTask1.py","DICTask2.py","DICTask3.py" input image path is the model output "VIZ" path.For example, if use sequence 1 the path is  
	"/deepwater-master/datasets/DIC-C2DH-HeLa/01_VIZ"
	
	We also provide sample result videos in the directory.
	
	
