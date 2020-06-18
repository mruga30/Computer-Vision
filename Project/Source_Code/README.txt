This file contains information to run the code:

1. The code was developed in pycharm on a Windows machine.
2. The packages required to run the code are: imageio, sys, numpy, scipy.misc, cv2, scipy.integrate, PIL
3. The list of files in our zip folder are: main.py, panorama3.py and make_cylindrical.py
4. To run these files:

main.py : 
cd into Source_Code and run:
python main.py -i <input_img> -x <x_window> -y <y_window> -w <window_width>
where i is the name of input image, x is the x center of window, y is the y center of window, w is the window width
This will produce an output file called extracted_output.jpg that contains the corneal extraction.

After running main.py on all eye images:
Create a list of files in the text document that we want to turn cylindrical and name it input_cylindrical.txt
Next, use the params.txt file as given in the submitted zipped folder
Next, open the terminal from bottom left in pycharm and run "make_cylindrical.py input_cylindrical.txt params.txt"
This gives a list of resulting images that are cylindrical

panorama.py : 
Open the main file in Pycharm
Have the input images in the folder as the panorama file and give the name accordingly in the file where it reads the images
Right click on the panorama.py file in the directory on the left and click run 'panorama.py'