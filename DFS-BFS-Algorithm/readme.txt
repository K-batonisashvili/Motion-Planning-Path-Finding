# Instructions

Within the zip file you shall find several pictures in PNG and JPG format. These images are used as tests for configuration space creation. Included are several GIF's which show segment tracing completed by main_results.py for their relevant environment.
main_results.py is the main python file which is used to generate the skeletonized form of the provided image, and then create a path to cover all possible segments.
The DEFAULT image is Soccer.jpg and can be found towards the bottom of the code. You are able to replace this image with any of the provided environments or you can upload a custom environment for further testing. 
Please ensure if you are uploading a custom image, it satisfies the requirements below.

Note: Pillow library is used for the sole purpose of having a visual output. Pillow requirement is not necessary if you are not planning on saving a GIF as an output. 




# Custom Image Requirements

	White Background with Black Lines
	JPG, JPEG, PNG, Format
	Image is store in the same directory as REST MODE.py
	image_path is changed appropriately within REST MODE.py

# Python Library Requirements

	python===3.12
	opencv-python==4.10.0.84
	numpy==2.1.1
	scikit-image==0.24.0
	pillow==10.2.0
	